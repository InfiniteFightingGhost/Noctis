from __future__ import annotations

import pytest

from app.training.config import training_config_from_payload


def test_training_config_parses_cnn_bilstm_payload() -> None:
    config = training_config_from_payload(
        {
            "dataset_dir": "out/dataset",
            "output_root": "models",
            "model_type": "cnn_bilstm",
            "feature_strategy": "sequence",
            "model": {
                "use_dataset_conditioning": True,
                "conditioning_mode": "embedding",
                "conditioning_embed_dim": 12,
                "conv_channels": [32, 64],
                "lstm_hidden_size": 128,
                "lstm_layers": 2,
            },
            "training": {
                "loss_type": "weighted_ce",
                "batch_size": 16,
                "max_epochs": 40,
                "early_stopping_patience": 6,
                "instability_macro_f1_threshold": 0.2,
            },
        }
    )
    assert config.model_type == "cnn_bilstm"
    assert config.model.use_dataset_conditioning is True
    assert config.model.conditioning_mode == "embedding"
    assert config.training.loss_type == "weighted_ce"
    assert config.training.batch_size == 16
    assert config.training.enable_binary_pretraining is True
    assert config.training.class_weight_strategy == "log_balanced"


def test_training_config_rejects_invalid_conditioning_mode() -> None:
    with pytest.raises(ValueError):
        training_config_from_payload(
            {
                "dataset_dir": "out/dataset",
                "output_root": "models",
                "model_type": "cnn_bilstm",
                "feature_strategy": "sequence",
                "model": {
                    "conditioning_mode": "invalid",
                },
            }
        )


def test_training_config_keeps_gradient_boosting_compatible() -> None:
    config = training_config_from_payload(
        {
            "dataset_dir": "out/dataset",
            "output_root": "models",
            "feature_schema_path": "out/dataset/feature_schema.json",
            "model_type": "gradient_boosting",
            "feature_strategy": "mean",
        }
    )
    assert config.model_type == "gradient_boosting"
    assert config.feature_strategy == "mean"


def test_training_config_parses_evaluation_policy_overrides() -> None:
    config = training_config_from_payload(
        {
            "dataset_dir": "out/dataset",
            "output_root": "models",
            "model_type": "cnn_bilstm",
            "feature_strategy": "sequence",
            "evaluation": {
                "calibration_bins": 10,
                "epoch_seconds": 20,
                "forbidden_transitions": [[0, 4]],
            },
        }
    )
    assert config.evaluation.calibration_bins == 10
    assert config.evaluation.epoch_seconds == 20
    assert config.evaluation.forbidden_transitions == [(0, 4)]


def test_training_config_accepts_focal_and_grouped_stratification() -> None:
    config = training_config_from_payload(
        {
            "dataset_dir": "out/dataset",
            "output_root": "models",
            "model_type": "cnn_bilstm",
            "feature_strategy": "sequence",
            "split_grouped_stratification": True,
            "split_stratify_key": "dataset",
            "training": {
                "loss_type": "focal",
                "focal_fallback_enabled": True,
                "pretrain_epochs": 8,
            },
        }
    )
    assert config.training.loss_type == "focal"
    assert config.split_grouped_stratification is True
    assert config.split_stratify_key == "dataset"


def test_training_config_parses_transition_and_crf_overrides() -> None:
    config = training_config_from_payload(
        {
            "dataset_dir": "out/dataset",
            "output_root": "models",
            "model_type": "cnn_bilstm",
            "feature_strategy": "sequence",
            "model": {
                "head_type": "crf",
                "crf_learn_transitions": True,
                "crf_init_from_priors": True,
                "crf_transition_l2": 1e-3,
            },
            "training": {
                "transition_reg_enabled": True,
                "transition_reg_lambda": 0.15,
                "transition_reg_mode": "prior_matrix",
                "persistence_reg_lambda": 0.05,
                "worst_night_sampling_enabled": True,
                "worst_night_fraction": 0.2,
            },
        }
    )
    assert config.model.head_type == "crf"
    assert config.model.crf_learn_transitions is True
    assert config.model.crf_init_from_priors is True
    assert config.model.crf_transition_l2 == pytest.approx(1e-3)
    assert config.training.transition_reg_enabled is True
    assert config.training.transition_reg_lambda == pytest.approx(0.15)
    assert config.training.transition_reg_mode == "prior_matrix"
    assert config.training.persistence_reg_lambda == pytest.approx(0.05)
    assert config.training.worst_night_sampling_enabled is True
    assert config.training.worst_night_fraction == pytest.approx(0.2)
