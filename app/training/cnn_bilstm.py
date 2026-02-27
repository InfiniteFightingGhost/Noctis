from __future__ import annotations

import json
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from app.training.config import TrainingConfig


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
        self._nn = nn
        if model_cfg is None:
            if config is None:
                raise ValueError("config or model_cfg is required")
            model_cfg = config.model

        def cfg(name: str):
            if isinstance(model_cfg, dict):
                return model_cfg[name]
            return getattr(model_cfg, name)

        self.num_domains = max(num_domains, 1)
        conv_layers: list[Any] = []
        in_channels = input_dim
        for channels in cfg("conv_channels"):
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=cfg("conv_kernel_size"),
                    padding=cfg("conv_kernel_size") // 2,
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(cfg("conv_dropout")))
            in_channels = channels
        self.conv = nn.Sequential(*conv_layers)
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
        lstm_input_dim = in_channels + conditioning_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=cfg("lstm_hidden_size"),
            num_layers=cfg("lstm_layers"),
            batch_first=True,
            dropout=cfg("lstm_dropout") if cfg("lstm_layers") > 1 else 0.0,
            bidirectional=True,
        )
        head_layers: list[Any] = []
        head_in = int(cfg("lstm_hidden_size")) * 2
        for hidden in cfg("head_hidden_dims"):
            head_layers.append(nn.Linear(head_in, hidden))
            head_layers.append(nn.ReLU())
            head_layers.append(nn.Dropout(cfg("head_dropout")))
            head_in = hidden
        head_layers.append(nn.Linear(head_in, num_classes))
        self.head = nn.Sequential(*head_layers)
        self._module = nn.Module()
        self._module.add_module("conv", self.conv)
        if self.domain_embedding is not None:
            self._module.add_module("domain_embedding", self.domain_embedding)
        self._module.add_module("lstm", self.lstm)
        self._module.add_module("head", self.head)

    @property
    def module(self):
        return self._module

    def __call__(self, x, domain_idx):
        torch = self._torch
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        if self.use_conditioning:
            if self.conditioning_mode == "embedding" and self.domain_embedding is not None:
                domain = self.domain_embedding(domain_idx)
            else:
                domain = torch.nn.functional.one_hot(
                    domain_idx,
                    num_classes=self.num_domains,
                ).float()
            domain = domain.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, domain], dim=-1)
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1)
        return self.head(pooled)


def train_cnn_bilstm(
    *,
    config: TrainingConfig,
    artifact_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    label_map: list[str],
    dataset_ids: np.ndarray,
    splits: dict[str, np.ndarray | None],
    evaluation_split_name: str,
) -> CnnBiLstmTrainOutput:
    torch = importlib.import_module("torch")
    nn = torch.nn

    if X.ndim != 3:
        raise ValueError("cnn_bilstm expects 3D sequence input")
    train_idx = splits.get("train")
    val_idx = splits.get("val")
    eval_idx = splits.get(evaluation_split_name)
    if train_idx is None or eval_idx is None:
        raise ValueError("Missing train or evaluation split")
    if len(train_idx) == 0:
        raise ValueError("Training split is empty")
    if len(eval_idx) == 0:
        raise ValueError("Evaluation split is empty")
    if val_idx is None or len(val_idx) == 0:
        val_idx = eval_idx

    y_lookup = {label: idx for idx, label in enumerate(label_map)}
    y_indices = np.asarray([y_lookup[str(label)] for label in y], dtype=np.int64)

    domain_values = [str(value) for value in dataset_ids]
    domain_lookup = {"DODH": 0, "CAP": 1, "ISRUC": 2, "UNKNOWN": 3}
    domain_indices = np.asarray(
        [domain_lookup.get(value, domain_lookup["UNKNOWN"]) for value in domain_values],
        dtype=np.int64,
    )

    X_norm, scaler_payload = _normalize_with_train_stats(X, train_idx)
    _ensure_finite("X_normalized", X_norm)

    model = CnnBiLstmNetwork(
        input_dim=int(X_norm.shape[2]),
        num_classes=len(label_map),
        num_domains=len(domain_lookup),
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
    scheduler = None
    if config.training.scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            min_lr=config.training.scheduler_min_lr,
        )
    class_weights, class_weight_adjustment = _build_class_weights(
        y_indices,
        dataset_ids,
        train_idx,
        label_map,
        n2_multiplier=config.training.n2_class_weight_multiplier,
    )
    criterion = _build_loss(config, class_weights, torch, nn)
    criterion = criterion.to(device)

    train_loader = _build_loader(
        X_norm, y_indices, domain_indices, train_idx, config.training.batch_size
    )
    val_loader = _build_loader(
        X_norm, y_indices, domain_indices, val_idx, config.training.batch_size
    )

    best_metric = float("-inf")
    stale_epochs = 0
    best_state: dict[str, Any] | None = None
    history_path = artifact_dir / "training_history.jsonl"

    for epoch in range(1, config.training.max_epochs + 1):
        train_loss = _run_epoch(
            net,
            model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=config.training.gradient_clip_norm,
            train=True,
        )
        val_loss, val_pred, val_true = _evaluate_epoch(
            net,
            model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        val_macro_f1 = float(f1_score(val_true, val_pred, average="macro"))
        if scheduler is not None:
            scheduler.step(val_macro_f1)
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history_row) + "\n")
        improved = val_macro_f1 > (best_metric + config.training.early_stopping_min_delta)
        if improved:
            best_metric = val_macro_f1
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in net.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs > config.training.early_stopping_patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    eval_loader = _build_loader(
        X_norm,
        y_indices,
        domain_indices,
        eval_idx,
        config.training.batch_size,
        shuffle=False,
    )
    _, eval_pred, eval_true = _evaluate_epoch(
        net,
        model,
        loader=eval_loader,
        criterion=criterion,
        device=device,
    )

    torch.save(net.state_dict(), artifact_dir / "model.pt")
    (artifact_dir / "scaler.json").write_text(json.dumps(scaler_payload, indent=2))

    metrics = _evaluate_global_metrics(eval_true, eval_pred, label_map)
    metrics["evaluation_split"] = evaluation_split_name

    eval_domains = dataset_ids[eval_idx]
    per_dataset = _evaluate_per_dataset_metrics(eval_true, eval_pred, eval_domains, label_map)
    per_dataset = _attach_dataset_aggregate(per_dataset)
    instability = _compute_instability_flags(
        per_dataset,
        threshold=config.training.instability_macro_f1_threshold,
    )
    class_distribution = _class_distribution_report(
        y=y,
        dataset_ids=dataset_ids,
        label_map=label_map,
        class_weights=class_weights,
        class_weight_adjustment=class_weight_adjustment,
    )
    (artifact_dir / "per_dataset_metrics.json").write_text(json.dumps(per_dataset, indent=2))
    (artifact_dir / "instability_flags.json").write_text(json.dumps(instability, indent=2))
    (artifact_dir / "class_distribution_report.json").write_text(
        json.dumps(class_distribution, indent=2)
    )

    if config.training.enable_domain_transfer_tests:
        transfer = _run_domain_transfer_tests(
            X=X_norm,
            y_indices=y_indices,
            dataset_ids=dataset_ids,
            label_map=label_map,
            config=config,
            class_weights=class_weights,
        )
    else:
        transfer = {
            "enabled": False,
            "runs": [],
            "mean_transfer_macro_f1": 0.0,
            "std_transfer_macro_f1": 0.0,
        }
    (artifact_dir / "domain_transfer_report.json").write_text(json.dumps(transfer, indent=2))

    return CnnBiLstmTrainOutput(
        metrics=metrics,
        model_classes=label_map,
        per_dataset_metrics=per_dataset,
    )


def _build_loader(
    X: np.ndarray,
    y: np.ndarray,
    domain_idx: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool = True,
):
    torch = importlib.import_module("torch")
    data = importlib.import_module("torch.utils.data")
    DataLoader = data.DataLoader
    TensorDataset = data.TensorDataset

    dataset = TensorDataset(
        torch.from_numpy(X[indices]).float(),
        torch.from_numpy(y[indices]).long(),
        torch.from_numpy(domain_idx[indices]).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    net,
    model: CnnBiLstmNetwork,
    *,
    loader,
    criterion,
    optimizer,
    device,
    gradient_clip_norm: float,
    train: bool,
) -> float:
    torch = importlib.import_module("torch")

    net.train(mode=train)
    total_loss = 0.0
    total_count = 0
    for batch_x, batch_y, batch_d in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_d = batch_d.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x, batch_d)
        loss = criterion(logits, batch_y)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
        batch_size = int(batch_x.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_count += batch_size
    if total_count == 0:
        return 0.0
    return total_loss / total_count


def _evaluate_epoch(net, model: CnnBiLstmNetwork, *, loader, criterion, device):
    torch = importlib.import_module("torch")

    net.eval()
    total_loss = 0.0
    total_count = 0
    pred: list[int] = []
    true: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y, batch_d in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_d = batch_d.to(device)
            logits = model(batch_x, batch_d)
            loss = criterion(logits, batch_y)
            probs = torch.argmax(logits, dim=1)
            pred.extend(probs.detach().cpu().numpy().tolist())
            true.extend(batch_y.detach().cpu().numpy().tolist())
            batch_size = int(batch_x.shape[0])
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_count += batch_size
    avg_loss = 0.0 if total_count == 0 else total_loss / total_count
    return avg_loss, np.asarray(pred, dtype=np.int64), np.asarray(true, dtype=np.int64)


def _build_loss(config: TrainingConfig, class_weights: np.ndarray, torch, nn):
    weight_tensor = torch.from_numpy(class_weights.astype(np.float32))
    ce = nn.CrossEntropyLoss(weight=weight_tensor)
    if config.training.loss_type == "weighted_ce":
        return ce

    gamma = config.training.focal_gamma

    class FocalLoss(nn.Module):
        def __init__(self, base_loss, focal_gamma: float) -> None:
            super().__init__()
            self.base_loss = base_loss
            self.focal_gamma = focal_gamma

        def forward(self, logits, target):
            ce_loss = nn.functional.cross_entropy(
                logits,
                target,
                weight=self.base_loss.weight,
                reduction="none",
            )
            prob_t = torch.exp(-ce_loss)
            focal = ((1.0 - prob_t) ** self.focal_gamma) * ce_loss
            return focal.mean()

    return FocalLoss(ce, gamma)


def _build_class_weights(
    y_indices: np.ndarray,
    dataset_ids: np.ndarray,
    train_idx: np.ndarray,
    label_map: list[str],
    *,
    n2_multiplier: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = y_indices[train_idx]
    unique, counts = np.unique(values, return_counts=True)
    weights = np.ones(len(label_map), dtype=np.float32)
    lookup = {int(k): int(v) for k, v in zip(unique, counts)}
    for idx in range(len(label_map)):
        count = lookup.get(idx, 0)
        if count > 0:
            weights[idx] = 1.0 / float(count)
    total = float(weights.sum())
    if total > 0:
        weights = weights * (len(label_map) / total)
    adjustment_log: dict[str, Any] = {
        "n2_adjusted": False,
        "n2_multiplier_applied": float(n2_multiplier),
        "reason": None,
        "triggered_datasets": [],
    }
    if "N2" in label_map:
        n2_index = label_map.index("N2")
        train_dataset_ids = np.asarray(dataset_ids[train_idx]).astype(str)
        triggered: list[str] = []
        for dataset_id in sorted(set(train_dataset_ids.tolist())):
            dataset_mask = train_dataset_ids == dataset_id
            if not dataset_mask.any():
                continue
            dataset_values = values[dataset_mask]
            proportion = float(np.mean(dataset_values == n2_index))
            if proportion > 0.55:
                triggered.append(dataset_id)
        if triggered:
            weights[n2_index] = weights[n2_index] * float(n2_multiplier)
            adjustment_log = {
                "n2_adjusted": True,
                "n2_multiplier_applied": float(n2_multiplier),
                "reason": "N2 proportion exceeded 0.55 in one or more datasets",
                "triggered_datasets": triggered,
            }
    return weights, adjustment_log


def _normalize_with_train_stats(
    X: np.ndarray, train_idx: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    train_values = X[train_idx]
    train_values = np.where(np.isfinite(train_values), train_values, np.nan)
    mean = np.nanmean(train_values, axis=(0, 1))
    std = np.nanstd(train_values, axis=(0, 1))
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    imputed = np.asarray(X, dtype=np.float32).copy()
    invalid = ~np.isfinite(imputed)
    if invalid.any():
        replacement = np.broadcast_to(mean.reshape(1, 1, -1), imputed.shape)
        imputed[invalid] = replacement[invalid]
    normalized = (imputed - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    payload = {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }
    return normalized.astype(np.float32), payload


def _ensure_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf after imputation")


def _evaluate_global_metrics(
    y_true_idx: np.ndarray, y_pred_idx: np.ndarray, label_map: list[str]
) -> dict[str, Any]:
    y_true = np.asarray([label_map[idx] for idx in y_true_idx], dtype=object)
    y_pred = np.asarray([label_map[idx] for idx in y_pred_idx], dtype=object)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class = _per_class_f1_scores(y_true, y_pred, label_map)
    matrix = confusion_matrix(y_true, y_pred, labels=label_map)
    return {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=label_map)),
        "per_class_f1": {label: float(score) for label, score in per_class.items()},
        "confusion_matrix": matrix.tolist(),
    }


def _per_class_f1_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: list[str],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label in label_map:
        y_true_binary = (y_true == label).astype(np.int32)
        y_pred_binary = (y_pred == label).astype(np.int32)
        scores[label] = float(f1_score(y_true_binary, y_pred_binary, average="binary"))
    return scores


def _evaluate_per_dataset_metrics(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    dataset_ids: np.ndarray,
    label_map: list[str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for dataset_id in sorted({str(value) for value in dataset_ids}):
        mask = np.asarray([str(value) == dataset_id for value in dataset_ids], dtype=bool)
        if not mask.any():
            continue
        metrics[dataset_id] = _evaluate_global_metrics(
            y_true_idx[mask],
            y_pred_idx[mask],
            label_map,
        )
        metrics[dataset_id]["samples"] = int(mask.sum())
    return metrics


def _compute_instability_flags(
    per_dataset_metrics: dict[str, Any], *, threshold: float
) -> dict[str, Any]:
    datasets_only = {
        key: value
        for key, value in per_dataset_metrics.items()
        if key not in {"aggregate"} and isinstance(value, dict)
    }
    macro_f1_values = [
        float(payload.get("macro_f1", 0.0))
        for payload in datasets_only.values()
        if isinstance(payload, dict)
    ]
    if not macro_f1_values:
        spread = 0.0
        lowest_dataset = None
        lowest_dataset_macro_f1 = None
    else:
        spread = float(max(macro_f1_values) - min(macro_f1_values))
        lowest_dataset = min(
            datasets_only,
            key=lambda item: float(datasets_only[item].get("macro_f1", 0.0)),
        )
        lowest_dataset_macro_f1 = float(datasets_only[lowest_dataset].get("macro_f1", 0.0))
    return {
        "threshold": float(threshold),
        "macro_f1_spread": spread,
        "DOMAIN_INSTABILITY": bool(spread > threshold),
        "instability_detected": bool(spread > threshold),
        "lowest_dataset": lowest_dataset,
        "lowest_dataset_macro_f1": lowest_dataset_macro_f1,
    }


def _class_distribution_report(
    *,
    y: np.ndarray,
    dataset_ids: np.ndarray,
    label_map: list[str],
    class_weights: np.ndarray,
    class_weight_adjustment: dict[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "datasets": {},
        "label_map": label_map,
        "class_weights": {label: float(class_weights[idx]) for idx, label in enumerate(label_map)},
        "class_weight_adjustments": class_weight_adjustment,
    }
    for dataset_id in sorted({str(value) for value in dataset_ids}):
        mask = np.asarray([str(value) == dataset_id for value in dataset_ids], dtype=bool)
        labels = y[mask]
        total = int(labels.shape[0])
        counts = {label: int(np.sum(labels == label)) for label in label_map}
        proportions = {
            "N1": float(counts.get("N1", 0) / total) if total else 0.0,
            "N2": float(counts.get("N2", 0) / total) if total else 0.0,
            "REM": float(counts.get("REM", 0) / total) if total else 0.0,
        }
        report["datasets"][dataset_id] = {
            "samples": total,
            "counts": counts,
            "proportions": proportions,
            "n2_downweight_applied": bool(
                class_weight_adjustment.get("n2_adjusted")
                and dataset_id in class_weight_adjustment.get("triggered_datasets", [])
            ),
        }
    return report


def _attach_dataset_aggregate(per_dataset_metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(per_dataset_metrics)
    macro_values = [
        float(payload.get("macro_f1", 0.0))
        for payload in metrics.values()
        if isinstance(payload, dict)
    ]
    if macro_values:
        metrics["aggregate"] = {
            "macro_f1_mean": float(np.mean(macro_values)),
            "macro_f1_std": float(np.std(macro_values)),
            "macro_f1_variance": float(np.var(macro_values)),
        }
    else:
        metrics["aggregate"] = {
            "macro_f1_mean": 0.0,
            "macro_f1_std": 0.0,
            "macro_f1_variance": 0.0,
        }
    return metrics


def _domain_transfer_combos() -> list[dict[str, Any]]:
    return [
        {
            "id": "train_DODH_ISRUC_test_CAP",
            "train": {"DODH", "ISRUC"},
            "test": "CAP",
        },
        {
            "id": "train_CAP_DODH_test_ISRUC",
            "train": {"CAP", "DODH"},
            "test": "ISRUC",
        },
        {
            "id": "train_CAP_ISRUC_test_DODH",
            "train": {"CAP", "ISRUC"},
            "test": "DODH",
        },
    ]


def _run_domain_transfer_tests(
    *,
    X: np.ndarray,
    y_indices: np.ndarray,
    dataset_ids: np.ndarray,
    label_map: list[str],
    config: TrainingConfig,
    class_weights: np.ndarray,
) -> dict[str, Any]:
    torch = importlib.import_module("torch")

    combos = _domain_transfer_combos()
    domain_values = np.asarray(dataset_ids).astype(str)
    domain_lookup = {"DODH": 0, "CAP": 1, "ISRUC": 2, "UNKNOWN": 3}
    domain_idx = np.asarray(
        [domain_lookup.get(name, domain_lookup["UNKNOWN"]) for name in domain_values],
        dtype=np.int64,
    )
    report_rows: list[dict[str, Any]] = []

    for combo in combos:
        train_mask = np.isin(domain_values, list(combo["train"]))
        test_mask = domain_values == combo["test"]
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        if train_indices.size == 0 or test_indices.size == 0:
            report_rows.append(
                {
                    "test_id": combo["id"],
                    "status": "skipped",
                    "reason": "missing_required_datasets",
                }
            )
            continue

        split = int(max(1, round(0.15 * train_indices.size)))
        val_indices = train_indices[-split:]
        train_indices = train_indices[:-split] if train_indices.size > split else train_indices
        if train_indices.size == 0:
            train_indices = val_indices

        model = CnnBiLstmNetwork(
            input_dim=int(X.shape[2]),
            num_classes=len(label_map),
            num_domains=len(domain_lookup),
            config=config,
            torch=torch,
            nn=torch.nn,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = model.module.to(device)
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(config.training.optimizer_beta1, config.training.optimizer_beta2),
        )
        criterion = _build_loss(config, class_weights, torch, torch.nn).to(device)

        train_loader = _build_loader(
            X, y_indices, domain_idx, train_indices, config.training.batch_size
        )
        val_loader = _build_loader(
            X, y_indices, domain_idx, val_indices, config.training.batch_size, shuffle=False
        )
        test_loader = _build_loader(
            X, y_indices, domain_idx, test_indices, config.training.batch_size, shuffle=False
        )

        max_epochs = min(config.training.max_epochs, 20)
        for _ in range(max_epochs):
            _run_epoch(
                net,
                model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                gradient_clip_norm=config.training.gradient_clip_norm,
                train=True,
            )

        _, val_pred, val_true = _evaluate_epoch(
            net, model, loader=val_loader, criterion=criterion, device=device
        )
        _, test_pred, test_true = _evaluate_epoch(
            net, model, loader=test_loader, criterion=criterion, device=device
        )
        val_macro_f1 = float(f1_score(val_true, val_pred, average="macro"))
        test_macro_f1 = float(f1_score(test_true, test_pred, average="macro"))
        test_metrics = _evaluate_global_metrics(test_true, test_pred, label_map)
        report_rows.append(
            {
                "test_id": combo["id"],
                "status": "completed",
                "train_datasets": sorted(combo["train"]),
                "heldout_dataset": combo["test"],
                "train_domain_f1": val_macro_f1,
                "heldout_domain_f1": test_macro_f1,
                "domain_transfer_gap": float(val_macro_f1 - test_macro_f1),
                "transfer_macro_f1": test_macro_f1,
                "transfer_per_class_f1": test_metrics["per_class_f1"],
                "transfer_confusion_matrix": test_metrics["confusion_matrix"],
                "transfer_balanced_accuracy": test_metrics["balanced_accuracy"],
                "transfer_cohen_kappa": test_metrics["kappa"],
            }
        )

    completed = [row for row in report_rows if row.get("status") == "completed"]
    f1_values = [float(row["transfer_macro_f1"]) for row in completed]
    return {
        "enabled": True,
        "runs": report_rows,
        "mean_transfer_macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "std_transfer_macro_f1": float(np.std(f1_values)) if f1_values else 0.0,
    }
