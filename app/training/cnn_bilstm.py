from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from app.eval import evaluate_all
from app.ml.decoding import transition_penalty_matrix, viterbi_decode_probabilities_with_penalties
from app.training.config import TrainingConfig
from app.training.mmwave import engineer_mmwave_features, low_agreement_threshold_from_train


@dataclass(frozen=True)
class CnnBiLstmTrainOutput:
    metrics: dict[str, Any]
    model_classes: list[str]
    per_dataset_metrics: dict[str, Any]


class CnnBiLstmNetwork:
    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        num_domains: int,
        config: TrainingConfig | None = None,
        model_cfg: Any | None = None,
        torch,
        nn,
    ) -> None:
        self._torch = torch
        if model_cfg is None:
            if config is None:
                raise ValueError("config or model_cfg is required")
            model_cfg = config.model

        def cfg(name: str):
            if isinstance(model_cfg, dict):
                return model_cfg.get(name)
            return getattr(model_cfg, name)

        self.num_domains = max(num_domains, 1)
        conv_blocks: list[Any] = []
        in_channels = input_dim
        self.use_residual_blocks = bool(cfg("use_residual_blocks"))
        dilation_schedule = list(cfg("dilation_schedule") or [1])
        for layer_idx, channels in enumerate(cfg("conv_channels")):
            dilation = int(dilation_schedule[min(layer_idx, len(dilation_schedule) - 1)])
            padding = (cfg("conv_kernel_size") // 2) * dilation
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=channels,
                        kernel_size=cfg("conv_kernel_size"),
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.ReLU(),
                    nn.Dropout(cfg("conv_dropout")),
                )
            )
            in_channels = channels
        self.conv_blocks = nn.ModuleList(conv_blocks)
        conditioning_dim = 0
        self.use_conditioning = bool(cfg("use_dataset_conditioning"))
        self.conditioning_mode = str(cfg("conditioning_mode"))
        if self.use_conditioning:
            if self.conditioning_mode == "embedding":
                self.domain_embedding = nn.Embedding(
                    num_domains, int(cfg("conditioning_embed_dim"))
                )
                conditioning_dim = int(cfg("conditioning_embed_dim"))
            else:
                self.domain_embedding = None
                conditioning_dim = self.num_domains
        else:
            self.domain_embedding = None
        self.lstm = nn.LSTM(
            input_size=in_channels + conditioning_dim,
            hidden_size=cfg("lstm_hidden_size"),
            num_layers=cfg("lstm_layers"),
            batch_first=True,
            dropout=cfg("lstm_dropout") if cfg("lstm_layers") > 1 else 0.0,
            bidirectional=True,
        )
        head_layers: list[Any] = []
        pooled_dim = int(cfg("lstm_hidden_size")) * 2
        head_in = pooled_dim
        for hidden in cfg("head_hidden_dims"):
            head_layers.append(nn.Linear(head_in, hidden))
            head_layers.append(nn.ReLU())
            head_layers.append(nn.Dropout(cfg("head_dropout")))
            head_in = hidden
        self.primary_head = nn.Sequential(*head_layers, nn.Linear(head_in, num_classes))
        self.aux_head = nn.Linear(pooled_dim, 1)

        self._module = nn.Module()
        self._module.add_module("conv_blocks", self.conv_blocks)
        if self.domain_embedding is not None:
            self._module.add_module("domain_embedding", self.domain_embedding)
        self._module.add_module("lstm", self.lstm)
        self._module.add_module("primary_head", self.primary_head)
        self._module.add_module("aux_head", self.aux_head)

    @property
    def module(self):
        return self._module

    def _forward_base(self, x, domain_idx):
        torch = self._torch
        x = x.transpose(1, 2)
        for block in self.conv_blocks:
            residual = x
            x = block(x)
            if self.use_residual_blocks and residual.shape == x.shape:
                x = x + residual
        x = x.transpose(1, 2)
        if self.use_conditioning:
            if self.conditioning_mode == "embedding" and self.domain_embedding is not None:
                domain = self.domain_embedding(domain_idx)
            else:
                domain = torch.nn.functional.one_hot(
                    domain_idx, num_classes=self.num_domains
                ).float()
            domain = domain.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, domain], dim=-1)
        lstm_out, _ = self.lstm(x)
        return lstm_out.mean(dim=1)

    def forward_with_aux(self, x, domain_idx):
        pooled = self._forward_base(x, domain_idx)
        return self.primary_head(pooled), self.aux_head(pooled).squeeze(-1)

    def __call__(self, x, domain_idx):
        logits, _ = self.forward_with_aux(x, domain_idx)
        return logits


def train_cnn_bilstm(
    *,
    config: TrainingConfig,
    artifact_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    label_map: list[str],
    feature_names: list[str],
    dataset_ids: np.ndarray,
    recording_ids: np.ndarray,
    splits: Mapping[str, np.ndarray | None],
    evaluation_split_name: str,
) -> CnnBiLstmTrainOutput:
    torch = importlib.import_module("torch")
    nn = torch.nn

    train_idx = splits.get("train")
    val_idx = splits.get("val")
    eval_idx = splits.get(evaluation_split_name)
    if train_idx is None or eval_idx is None:
        raise ValueError("Missing train or evaluation split")
    if val_idx is None or len(val_idx) == 0:
        val_idx = eval_idx

    threshold = low_agreement_threshold_from_train(
        X, feature_names=feature_names, train_indices=train_idx
    )
    X_eng, final_feature_names, formulas = engineer_mmwave_features(
        X,
        feature_names=feature_names,
        low_agreement_threshold=threshold,
    )
    X_norm, scaler_payload = _normalize_with_train_stats(
        X_eng,
        train_idx,
        recording_ids=recording_ids,
        policy=config.training.normalization_policy,
    )

    y_lookup = {label: idx for idx, label in enumerate(label_map)}
    y_indices = np.asarray([y_lookup[str(label)] for label in y], dtype=np.int64)
    wake_index = y_lookup["W"]
    dataset_map = {"DODH": 0, "CAP": 1, "ISRUC": 2, "SLEEP-EDF": 3, "UNKNOWN": 4}
    domain_indices = np.asarray(
        [dataset_map.get(str(x), dataset_map["UNKNOWN"]) for x in dataset_ids], dtype=np.int64
    )

    model = CnnBiLstmNetwork(
        input_dim=int(X_norm.shape[2]),
        num_classes=len(label_map),
        num_domains=len(dataset_map),
        config=config,
        torch=torch,
        nn=nn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.module.to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.optimizer_beta1, config.training.optimizer_beta2),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        min_lr=config.training.scheduler_min_lr,
    )
    class_weights = _build_class_weights(
        y_indices,
        train_idx,
        len(label_map),
        strategy=config.training.class_weight_strategy,
    )
    class_weight_tensor = torch.from_numpy(class_weights.astype(np.float32)).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weight_tensor)
    bce_loss = nn.BCEWithLogitsLoss()

    train_loader = _build_loader(
        X_norm, y_indices, domain_indices, train_idx, config.training.batch_size
    )
    train_eval_loader = _build_loader(
        X_norm, y_indices, domain_indices, train_idx, config.training.batch_size, shuffle=False
    )
    val_loader = _build_loader(
        X_norm, y_indices, domain_indices, val_idx, config.training.batch_size, shuffle=False
    )

    decode_transition_probs = _build_decode_transition_probabilities(
        y_indices=y_indices,
        recording_ids=recording_ids,
        train_idx=train_idx,
        label_map=label_map,
        learn_transitions=config.model.crf_learn_transitions or config.model.head_type == "crf",
        init_from_priors=config.model.crf_init_from_priors,
        l2_strength=config.model.crf_transition_l2,
    )
    decode_transition_penalties = _transition_penalties_from_probabilities(decode_transition_probs)

    best_metric = float("-inf")
    stale_epochs = 0
    best_state: dict[str, Any] | None = None
    use_focal = config.training.loss_type == "focal"
    rem_recall_stale = 0
    hard_mining_records: set[str] = set()
    history_path = artifact_dir / "training_history.jsonl"
    metrics_epoch_path = artifact_dir / "metrics_epoch.jsonl"

    if config.training.enable_binary_pretraining and config.training.pretrain_epochs > 0:
        if config.training.pretrain_freeze_primary_head:
            for param in model.primary_head.parameters():
                param.requires_grad = False
        pretrain_optimizer = torch.optim.AdamW(
            [p for p in net.parameters() if p.requires_grad],
            lr=config.training.learning_rate * config.training.pretrain_lr_scale,
            weight_decay=config.training.weight_decay,
            betas=(config.training.optimizer_beta1, config.training.optimizer_beta2),
        )
        for epoch in range(1, config.training.pretrain_epochs + 1):
            train_loss = _run_epoch(
                net,
                model,
                loader=train_loader,
                ce_loss=ce_loss,
                bce_loss=bce_loss,
                optimizer=pretrain_optimizer,
                device=device,
                gradient_clip_norm=config.training.gradient_clip_norm,
                wake_index=wake_index,
                aux_lambda=config.training.auxiliary_loss_lambda,
                train=True,
                loss_type="binary_pretrain",
                focal_gamma=config.training.focal_gamma,
                recording_ids=recording_ids,
                transition_penalties=decode_transition_penalties,
                transition_reg_enabled=False,
                transition_reg_lambda=0.0,
                persistence_reg_lambda=0.0,
            )
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "stage": "pretrain",
                            "train_loss": train_loss,
                            "learning_rate": float(pretrain_optimizer.param_groups[0]["lr"]),
                        }
                    )
                    + "\n"
                )
        if config.training.pretrain_freeze_primary_head:
            for param in model.primary_head.parameters():
                param.requires_grad = True

    for epoch in range(1, config.training.max_epochs + 1):
        if config.training.hard_mining_enabled and hard_mining_records:
            train_loader = _build_loader(
                X_norm,
                y_indices,
                domain_indices,
                train_idx,
                config.training.batch_size,
                recording_ids=recording_ids,
                hard_mining_records=hard_mining_records,
                hard_mining_oversample_factor=config.training.hard_mining_oversample_factor,
            )
        train_loss = _run_epoch(
            net,
            model,
            loader=train_loader,
            ce_loss=ce_loss,
            bce_loss=bce_loss,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=config.training.gradient_clip_norm,
            wake_index=wake_index,
            aux_lambda=config.training.auxiliary_loss_lambda,
            train=True,
            loss_type="focal" if use_focal else "weighted_ce",
            focal_gamma=config.training.focal_gamma,
            recording_ids=recording_ids,
            transition_penalties=decode_transition_penalties,
            transition_reg_enabled=config.training.transition_reg_enabled,
            transition_reg_lambda=config.training.transition_reg_lambda,
            persistence_reg_lambda=config.training.persistence_reg_lambda,
        )
        train_eval = _evaluate_epoch(
            net,
            model,
            loader=train_eval_loader,
            ce_loss=ce_loss,
            bce_loss=bce_loss,
            device=device,
            wake_index=wake_index,
            aux_lambda=config.training.auxiliary_loss_lambda,
            loss_type="focal" if use_focal else "weighted_ce",
            focal_gamma=config.training.focal_gamma,
            num_classes=len(label_map),
        )
        val = _evaluate_epoch(
            net,
            model,
            loader=val_loader,
            ce_loss=ce_loss,
            bce_loss=bce_loss,
            device=device,
            wake_index=wake_index,
            aux_lambda=config.training.auxiliary_loss_lambda,
            loss_type="focal" if use_focal else "weighted_ce",
            focal_gamma=config.training.focal_gamma,
            num_classes=len(label_map),
        )
        rem_recall = float(val["per_class_recall"].get("REM", 0.0))
        if config.training.focal_fallback_enabled and not use_focal:
            if rem_recall < config.training.focal_trigger_rem_recall:
                rem_recall_stale += 1
            else:
                rem_recall_stale = 0
            if rem_recall_stale >= config.training.focal_trigger_patience:
                use_focal = True
        scheduler.step(val["macro_f1"])
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "stage": "finetune",
                        "train_loss": train_loss,
                        "val_loss": val["loss"],
                        "val_macro_f1": val["macro_f1"],
                        "val_rem_recall": rem_recall,
                        "loss_mode": "focal" if use_focal else "weighted_ce",
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }
                )
                + "\n"
            )
        with metrics_epoch_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train": {
                            "loss": train_eval["loss"],
                            "macro_f1": train_eval["macro_f1"],
                        },
                        "val": {
                            "loss": val["loss"],
                            "macro_f1": val["macro_f1"],
                            "rem_recall": rem_recall,
                        },
                        "gap": {
                            "loss": float(val["loss"] - train_eval["loss"]),
                            "macro_f1": float(train_eval["macro_f1"] - val["macro_f1"]),
                        },
                    }
                )
                + "\n"
            )

        if config.training.hard_mining_enabled:
            hard_mining_records = _select_hard_mining_recordings(
                model,
                X_norm,
                y_indices,
                domain_indices,
                recording_ids,
                train_idx,
                device=device,
                fraction=config.training.hard_mining_fraction,
            )
        if val["macro_f1"] > best_metric + config.training.early_stopping_min_delta:
            best_metric = val["macro_f1"]
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs > config.training.early_stopping_patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    val_logits = _predict_logits(model, X_norm[val_idx], domain_indices[val_idx], device=device)
    temperature = _fit_temperature(val_logits, y_indices[val_idx])
    eval_logits = _predict_logits(model, X_norm[eval_idx], domain_indices[eval_idx], device=device)
    eval_probs = _softmax_with_temperature(eval_logits, temperature)
    pre_pred = np.argmax(eval_probs, axis=1)
    post_pred = _decode_by_recording(
        eval_probs,
        recording_ids[eval_idx],
        label_map,
        transition_penalties=decode_transition_penalties,
    )

    metrics_pre = _build_metrics(y_indices[eval_idx], pre_pred, eval_probs, label_map)
    metrics_post = _build_metrics(y_indices[eval_idx], post_pred, eval_probs, label_map)
    per_dataset = _per_dataset_breakdown(
        y_true=y_indices[eval_idx],
        pre_pred=pre_pred,
        post_pred=post_pred,
        probs=eval_probs,
        dataset_ids=dataset_ids[eval_idx],
        label_map=label_map,
    )
    spread = _macro_f1_spread(per_dataset)
    worst = _worst_dataset(per_dataset)
    metrics = {
        "evaluation_split": evaluation_split_name,
        "training": {
            "binary_pretraining_enabled": config.training.enable_binary_pretraining,
            "pretrain_epochs": int(config.training.pretrain_epochs),
            "focal_fallback_enabled": config.training.focal_fallback_enabled,
            "focal_activated": use_focal,
            "class_weight_strategy": config.training.class_weight_strategy,
            "hard_mining_enabled": config.training.hard_mining_enabled,
            "head_type": config.model.head_type,
            "crf_learn_transitions": config.model.crf_learn_transitions,
            "transition_reg_enabled": config.training.transition_reg_enabled,
            "transition_reg_lambda": config.training.transition_reg_lambda,
            "persistence_reg_lambda": config.training.persistence_reg_lambda,
        },
        "pre_decode": metrics_pre,
        "post_decode": metrics_post,
        "per_dataset": per_dataset,
        "worst_dataset_alert": {
            "triggered": spread > 0.15,
            "macro_f1_spread": spread,
            "threshold": 0.15,
            "worst_dataset": worst,
        },
    }
    metrics["temperature"] = temperature
    metrics["input_dim"] = int(X_norm.shape[2])
    metrics["class_distribution"] = _class_distribution(y_indices, label_map)
    metrics["feature_pipeline"] = {
        "base_feature_schema": feature_names,
        "engineered_features": final_feature_names[len(feature_names) :],
        "final_feature_schema": final_feature_names,
        "engineered_formulas": formulas,
        "eps": 1e-6,
        "low_agreement_threshold": threshold,
        "requires_flags": True,
        "requires_agree_flags": True,
    }

    scorecard = evaluate_all(
        y_true=y_indices[eval_idx],
        y_pred=post_pred,
        proba=eval_probs,
        recording_id=recording_ids[eval_idx],
        dataset_id=dataset_ids[eval_idx],
        class_names=label_map,
        epoch_seconds=config.evaluation.epoch_seconds,
        calibration_bins=config.evaluation.calibration_bins,
        forbidden_transitions=set(config.evaluation.forbidden_transitions),
        hard_thresholds=config.evaluation.hard_thresholds,
        soft_thresholds=config.evaluation.soft_thresholds,
    )
    support_total = max(int(scorecard["classification"]["global"].get("support_total", 0)), 1)
    confusion = np.asarray(
        scorecard["classification"]["global"]["confusion_matrix"], dtype=np.float64
    )
    raw_accuracy = float(np.trace(confusion) / support_total)
    scorecard["accuracy"] = raw_accuracy
    scorecard["macro_f1"] = float(scorecard["classification"]["global"]["macro_f1"])
    scorecard["evaluation_split"] = evaluation_split_name
    scorecard["temperature"] = temperature
    scorecard["input_dim"] = int(X_norm.shape[2])
    scorecard["feature_pipeline"] = metrics["feature_pipeline"]
    scorecard["pre_decode"] = metrics_pre
    scorecard["post_decode"] = metrics_post
    scorecard["per_dataset"] = per_dataset
    scorecard["worst_dataset_alert"] = metrics["worst_dataset_alert"]
    scorecard["training"] = metrics.get("training", {})
    scorecard["class_distribution"] = metrics.get("class_distribution", {})
    metrics = scorecard

    torch.save(net.state_dict(), artifact_dir / "model.pt")
    (artifact_dir / "scaler.json").write_text(json.dumps(scaler_payload, indent=2))
    (artifact_dir / "temperature.json").write_text(
        json.dumps({"temperature": temperature}, indent=2)
    )
    (artifact_dir / "transition_matrix.json").write_text(
        json.dumps(
            {
                "labels": label_map,
                "probabilities": decode_transition_probs.tolist(),
                "penalties": decode_transition_penalties.tolist(),
            },
            indent=2,
        )
    )
    (artifact_dir / "feature_pipeline.json").write_text(
        json.dumps(metrics["feature_pipeline"], indent=2)
    )
    (artifact_dir / "per_dataset_metrics.json").write_text(json.dumps(per_dataset, indent=2))
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (artifact_dir / "calibration.json").write_text(json.dumps(metrics["calibration"], indent=2))
    (artifact_dir / "night_metrics.json").write_text(json.dumps(metrics["night"], indent=2))
    (artifact_dir / "domain_metrics.json").write_text(json.dumps(metrics["domain"], indent=2))
    (artifact_dir / "robustness_report.json").write_text(
        json.dumps(metrics["robustness"], indent=2)
    )
    return CnnBiLstmTrainOutput(
        metrics=metrics, model_classes=label_map, per_dataset_metrics=per_dataset
    )


def _build_loader(
    X,
    y,
    domain_idx,
    indices,
    batch_size,
    *,
    shuffle=True,
    recording_ids: np.ndarray | None = None,
    hard_mining_records: set[str] | None = None,
    hard_mining_oversample_factor: float = 1.0,
):
    torch = importlib.import_module("torch")
    data = importlib.import_module("torch.utils.data")
    dataset = data.TensorDataset(
        torch.from_numpy(X[indices]).float(),
        torch.from_numpy(y[indices]).long(),
        torch.from_numpy(domain_idx[indices]).long(),
        torch.from_numpy(np.asarray(indices, dtype=np.int64)).long(),
    )
    if recording_ids is not None and hard_mining_records:
        weights = np.ones(len(indices), dtype=np.float32)
        for pos, idx in enumerate(indices):
            if str(recording_ids[int(idx)]) in hard_mining_records:
                weights[pos] = float(hard_mining_oversample_factor)
        sampler = data.WeightedRandomSampler(
            torch.from_numpy(weights), num_samples=len(weights), replacement=True
        )
        return data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=bool(shuffle))


def _run_epoch(
    net,
    model,
    *,
    loader,
    ce_loss,
    bce_loss,
    optimizer,
    device,
    gradient_clip_norm: float,
    wake_index: int,
    aux_lambda: float,
    train: bool,
    loss_type: str,
    focal_gamma: float,
    recording_ids: np.ndarray,
    transition_penalties: np.ndarray,
    transition_reg_enabled: bool,
    transition_reg_lambda: float,
    persistence_reg_lambda: float,
) -> float:
    torch = importlib.import_module("torch")
    net.train(mode=train)
    total_loss = 0.0
    total_count = 0
    for batch_x, batch_y, batch_d, batch_idx in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_d = batch_d.to(device)
        batch_idx_np = batch_idx.detach().cpu().numpy()
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits, aux_logits = model.forward_with_aux(batch_x, batch_d)
        wake_target = (batch_y == wake_index).float()
        if loss_type == "binary_pretrain":
            class_loss = 0.0 * logits.mean()
        elif loss_type == "focal":
            class_loss = _focal_loss(logits, batch_y, ce_loss.weight, gamma=focal_gamma)
        else:
            class_loss = ce_loss(logits, batch_y)
        loss = class_loss + aux_lambda * bce_loss(aux_logits, wake_target)
        if train and transition_reg_enabled and loss_type != "binary_pretrain":
            probs = torch.softmax(logits, dim=1)
            trans_loss, persist_loss = _batch_transition_regularization(
                probs=probs,
                batch_indices=batch_idx_np,
                recording_ids=recording_ids,
                penalties=transition_penalties,
                torch=torch,
            )
            loss = loss + transition_reg_lambda * trans_loss + persistence_reg_lambda * persist_loss
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
        size = int(batch_x.shape[0])
        total_loss += float(loss.detach().cpu().item()) * size
        total_count += size
    return 0.0 if total_count == 0 else total_loss / total_count


def _evaluate_epoch(
    net,
    model,
    *,
    loader,
    ce_loss,
    bce_loss,
    device,
    wake_index: int,
    aux_lambda: float,
    loss_type: str,
    focal_gamma: float,
    num_classes: int,
):
    torch = importlib.import_module("torch")
    net.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y, batch_d, _batch_idx in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_d = batch_d.to(device)
            logits, aux_logits = model.forward_with_aux(batch_x, batch_d)
            wake_target = (batch_y == wake_index).float()
            if loss_type == "focal":
                class_loss = _focal_loss(logits, batch_y, ce_loss.weight, gamma=focal_gamma)
            else:
                class_loss = ce_loss(logits, batch_y)
            loss = class_loss + aux_lambda * bce_loss(aux_logits, wake_target)
            losses.append(float(loss.detach().cpu().item()))
            y_true.extend(batch_y.detach().cpu().numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
    macro = (
        float(f1_score(np.asarray(y_true), np.asarray(y_pred), average="macro")) if y_true else 0.0
    )
    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=np.arange(num_classes))
    denom = np.sum(cm, axis=1)
    per_class = np.divide(
        np.diag(cm).astype(np.float64),
        denom.astype(np.float64),
        out=np.zeros(num_classes, dtype=np.float64),
        where=denom > 0,
    )
    recall_map = {
        "W": float(per_class[0]),
        "Light": float(per_class[1]),
        "Deep": float(per_class[2]),
        "REM": float(per_class[3]),
    }
    return {
        "loss": float(np.mean(losses) if losses else 0.0),
        "macro_f1": macro,
        "per_class_recall": recall_map,
    }


def _predict_logits(model, X: np.ndarray, domain_idx: np.ndarray, *, device) -> np.ndarray:
    torch = importlib.import_module("torch")
    model.module.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(X).float().to(device),
            torch.from_numpy(domain_idx).long().to(device),
        )
    return logits.detach().cpu().numpy()


def _fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    torch = importlib.import_module("torch")
    t = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=50)
    logits_t = torch.from_numpy(logits).float()
    targets = torch.from_numpy(y_true).long()
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = criterion(logits_t / torch.clamp(t, min=0.05), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(t.detach(), min=0.05).item())


def _softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exps = np.exp(scaled)
    return exps / np.sum(exps, axis=1, keepdims=True)


def _decode_by_recording(
    probs: np.ndarray,
    recording_ids: np.ndarray,
    label_map: list[str],
    *,
    transition_penalties: np.ndarray | None,
) -> np.ndarray:
    pred = np.zeros(probs.shape[0], dtype=np.int64)
    for rec_id in sorted({str(x) for x in recording_ids}):
        mask = np.asarray([str(x) == rec_id for x in recording_ids], dtype=bool)
        pred[mask] = viterbi_decode_probabilities_with_penalties(
            probs[mask],
            label_map,
            transition_penalties=transition_penalties,
        )
    return pred


def _build_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, label_map: list[str]
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_map)))
    ece = _expected_calibration_error(y_true, probs)
    per_class_f1: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    for idx, label in enumerate(label_map):
        true_bin = (y_true == idx).astype(np.int32)
        pred_bin = (y_pred == idx).astype(np.int32)
        per_class_f1[label] = float(f1_score(true_bin, pred_bin, average="binary"))
        per_class_recall[label] = float(recall_score(true_bin, pred_bin, average="binary"))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=np.arange(len(label_map)))),
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "ece": ece,
    }


def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    if probs.size == 0:
        return 0.0
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    total = conf.shape[0]
    error = 0.0
    for idx in range(bins):
        mask = bucket == idx
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(np.mean(conf[mask]))
        avg_acc = float(np.mean(correct[mask]))
        error += abs(avg_conf - avg_acc) * (count / total)
    return float(error)


def _per_dataset_breakdown(
    *,
    y_true: np.ndarray,
    pre_pred: np.ndarray,
    post_pred: np.ndarray,
    probs: np.ndarray,
    dataset_ids: np.ndarray,
    label_map: list[str],
) -> dict[str, Any]:
    requested = ["CAP", "ISRUC", "SLEEP-EDF"]
    out: dict[str, Any] = {}
    for dataset_id in requested:
        mask = np.asarray([str(v) == dataset_id for v in dataset_ids], dtype=bool)
        if not mask.any():
            out[dataset_id] = {"samples": 0}
            continue
        out[dataset_id] = {
            "samples": int(mask.sum()),
            "pre_decode": _build_metrics(y_true[mask], pre_pred[mask], probs[mask], label_map),
            "post_decode": _build_metrics(y_true[mask], post_pred[mask], probs[mask], label_map),
        }
    return out


def _macro_f1_spread(per_dataset: dict[str, Any]) -> float:
    values = [
        float(v["post_decode"]["macro_f1"])
        for v in per_dataset.values()
        if isinstance(v, dict) and v.get("samples", 0) > 0 and "post_decode" in v
    ]
    if not values:
        return 0.0
    return float(max(values) - min(values))


def _worst_dataset(per_dataset: dict[str, Any]) -> str | None:
    candidates = [
        (k, float(v["post_decode"]["macro_f1"]))
        for k, v in per_dataset.items()
        if isinstance(v, dict) and v.get("samples", 0) > 0 and "post_decode" in v
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[1])[0][0]


def _normalize_with_train_stats(
    X: np.ndarray,
    train_idx: np.ndarray,
    *,
    recording_ids: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    normalized = np.asarray(X, dtype=np.float32).copy()
    if policy == "recording_zscore_then_global":
        for recording_id in sorted({str(v) for v in recording_ids}):
            mask = np.asarray([str(v) == recording_id for v in recording_ids], dtype=bool)
            values = np.where(np.isfinite(normalized[mask]), normalized[mask], np.nan)
            local_mean = np.nanmean(values, axis=(0, 1))
            local_std = np.nanstd(values, axis=(0, 1))
            local_mean = np.where(np.isfinite(local_mean), local_mean, 0.0)
            local_std = np.where(np.isfinite(local_std) & (local_std > 1e-8), local_std, 1.0)
            replacement = np.broadcast_to(local_mean.reshape(1, 1, -1), normalized[mask].shape)
            rec_values = normalized[mask]
            rec_missing = ~np.isfinite(rec_values)
            if rec_missing.any():
                rec_values = rec_values.copy()
                rec_values[rec_missing] = replacement[rec_missing]
            normalized[mask] = (rec_values - local_mean.reshape(1, 1, -1)) / local_std.reshape(
                1, 1, -1
            )

    train_values = np.where(np.isfinite(normalized[train_idx]), normalized[train_idx], np.nan)
    mean = np.nanmean(train_values, axis=(0, 1))
    std = np.nanstd(train_values, axis=(0, 1))
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    filled = normalized
    mask = ~np.isfinite(filled)
    if mask.any():
        replacement = np.broadcast_to(mean.reshape(1, 1, -1), filled.shape)
        filled = filled.copy()
        filled[mask] = replacement[mask]
    normalized = (filled - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return normalized.astype(np.float32), {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "policy": policy,
    }


def _build_class_weights(
    y_indices: np.ndarray,
    train_idx: np.ndarray,
    num_classes: int,
    *,
    strategy: str,
) -> np.ndarray:
    values = y_indices[train_idx]
    unique, counts = np.unique(values, return_counts=True)
    lookup = {int(k): int(v) for k, v in zip(unique, counts, strict=True)}
    weights = np.ones(num_classes, dtype=np.float32)
    for idx in range(num_classes):
        count = lookup.get(idx, 0)
        if count > 0:
            freq = float(count) / max(float(values.shape[0]), 1.0)
            if strategy == "log_balanced":
                weights[idx] = 1.0 / float(np.log(1.02 + freq))
            else:
                weights[idx] = 1.0 / float(count)
    total = float(weights.sum())
    return weights * (num_classes / total)


def _focal_loss(logits, targets, class_weights, *, gamma: float):
    torch = importlib.import_module("torch")
    ce = torch.nn.functional.cross_entropy(
        logits,
        targets,
        reduction="none",
        weight=class_weights,
    )
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


def _select_hard_mining_recordings(
    model,
    X: np.ndarray,
    y: np.ndarray,
    domain_indices: np.ndarray,
    recording_ids: np.ndarray,
    train_idx: np.ndarray,
    *,
    device,
    fraction: float,
) -> set[str]:
    if fraction <= 0:
        return set()
    logits = _predict_logits(model, X[train_idx], domain_indices[train_idx], device=device)
    pred = np.argmax(logits, axis=1)
    scores: list[tuple[str, float]] = []
    unique_records = sorted({str(v) for v in recording_ids[train_idx]})
    for rec in unique_records:
        mask = np.asarray([str(v) == rec for v in recording_ids[train_idx]], dtype=bool)
        if not mask.any():
            continue
        rec_f1 = float(f1_score(y[train_idx][mask], pred[mask], average="macro"))
        scores.append((rec, rec_f1))
    if not scores:
        return set()
    scores.sort(key=lambda item: item[1])
    count = max(int(np.ceil(len(scores) * fraction)), 1)
    return {rec for rec, _ in scores[:count]}


def _class_distribution(y_indices: np.ndarray, label_map: list[str]) -> dict[str, float]:
    total = max(int(y_indices.shape[0]), 1)
    out: dict[str, float] = {}
    for idx, label in enumerate(label_map):
        out[str(label)] = float((y_indices == idx).sum() / total)
    return out


def _build_decode_transition_probabilities(
    *,
    y_indices: np.ndarray,
    recording_ids: np.ndarray,
    train_idx: np.ndarray,
    label_map: list[str],
    learn_transitions: bool,
    init_from_priors: bool,
    l2_strength: float,
) -> np.ndarray:
    n = len(label_map)
    base_penalties = transition_penalty_matrix(label_map)
    base_probs = np.exp(-base_penalties.astype(np.float64))
    base_probs = base_probs / np.clip(base_probs.sum(axis=1, keepdims=True), 1e-12, None)
    if not init_from_priors:
        base_probs = np.full((n, n), 1.0 / n, dtype=np.float64)

    if not learn_transitions:
        return base_probs.astype(np.float32)

    counts = np.zeros((n, n), dtype=np.float64)
    train_sorted = np.asarray(sorted(int(idx) for idx in train_idx), dtype=np.int64)
    for idx in range(len(train_sorted) - 1):
        src_idx = int(train_sorted[idx])
        dst_idx = int(train_sorted[idx + 1])
        if str(recording_ids[src_idx]) != str(recording_ids[dst_idx]):
            continue
        src = int(y_indices[src_idx])
        dst = int(y_indices[dst_idx])
        counts[src, dst] += 1.0
    empirical = counts + 1.0
    empirical = empirical / np.clip(empirical.sum(axis=1, keepdims=True), 1e-12, None)
    alpha = 1.0 / (1.0 + max(l2_strength, 0.0) * 1000.0)
    mixed = alpha * empirical + (1.0 - alpha) * base_probs
    mixed = mixed / np.clip(mixed.sum(axis=1, keepdims=True), 1e-12, None)
    return mixed.astype(np.float32)


def _transition_penalties_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(probs, 1e-8, 1.0)
    penalties = -np.log(probs)
    return penalties.astype(np.float32)


def _batch_transition_regularization(
    *,
    probs,
    batch_indices: np.ndarray,
    recording_ids: np.ndarray,
    penalties: np.ndarray,
    torch,
):
    n = int(batch_indices.shape[0])
    if n <= 1:
        zero = probs.sum() * 0.0
        return zero, zero
    order = np.argsort(batch_indices)
    pairs: list[tuple[int, int]] = []
    for pos in range(n - 1):
        left_local = int(order[pos])
        right_local = int(order[pos + 1])
        left_global = int(batch_indices[left_local])
        right_global = int(batch_indices[right_local])
        if right_global != left_global + 1:
            continue
        if str(recording_ids[left_global]) != str(recording_ids[right_global]):
            continue
        pairs.append((left_local, right_local))
    if not pairs:
        zero = probs.sum() * 0.0
        return zero, zero

    i_idx = torch.tensor([left for left, _ in pairs], device=probs.device)
    j_idx = torch.tensor([right for _, right in pairs], device=probs.device)
    p_i = probs.index_select(0, i_idx)
    p_j = probs.index_select(0, j_idx)
    penalty_tensor = torch.from_numpy(np.asarray(penalties, dtype=np.float32)).to(probs.device)
    expected = torch.einsum("bi,ij,bj->b", p_i, penalty_tensor, p_j)
    transition_loss = expected.mean()
    persistence_loss = (1.0 - (p_i * p_j).sum(dim=1)).mean()
    return transition_loss, persistence_loss
