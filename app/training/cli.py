from __future__ import annotations

import argparse
from pathlib import Path

from app.training.trainer import TrainingConfig, persist_model, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Noctis training pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a linear softmax model")
    train_parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing dataset.npz",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write model artifacts",
    )
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--learning-rate", type=float, default=0.1)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--l2-weight", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--window-aggregation",
        type=str,
        default="mean",
        choices=["mean"],
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        config = TrainingConfig(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            l2_weight=args.l2_weight,
            seed=args.seed,
            window_aggregation=args.window_aggregation,
        )
        result = train_model(config)
        persist_model(result, config.dataset_dir, config.output_dir)


if __name__ == "__main__":
    main()
