"""Training utilities for fraud detection."""
from __future__ import annotations

import argparse
from pathlib import Path

def train(input_path: str, epochs: int, lr: float, outdir: str) -> str:
    """Dummy training routine.

    Args:
        input_path: Path to training data.
        epochs: Number of epochs to train for.
        lr: Learning rate.
        outdir: Directory to write outputs to.

    Returns:
        Path to a file representing the trained model.
    """
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.txt"
    with model_path.open("w", encoding="utf-8") as f:
        f.write(f"trained on {input_path} for {epochs} epochs with lr {lr}\n")
    return str(model_path)


def main(argv: list[str] | None = None) -> None:
    """Command line interface for training.

    Parses command line arguments and invokes :func:`train`.
    """
    parser = argparse.ArgumentParser(description="Train a fraud detection model")
    parser.add_argument("--input", required=True, help="Path to input data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--outdir", default=".", help="Directory to place outputs")
    args = parser.parse_args(argv)
    train(args.input, args.epochs, args.lr, args.outdir)


if __name__ == "__main__":
    main()
