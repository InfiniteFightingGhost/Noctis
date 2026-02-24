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
                "lstm_hidden_size": 64,
            },
            "training": {
                "loss_type": "focal",
                "batch_size": 16,
                "instability_macro_f1_threshold": 0.2,
            },
        }
    )
    assert config.model_type == "cnn_bilstm"
    assert config.model.use_dataset_conditioning is True
    assert config.model.conditioning_mode == "embedding"
    assert config.training.loss_type == "focal"
    assert config.training.batch_size == 16


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
