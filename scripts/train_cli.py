#!/usr/bin/env python3
"""Command line interface to train the fraud model."""

import argparse
import pandas as pd

from fraud_model.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud model on a CSV dataset")
    parser.add_argument("csv", type=str, help="Path to CSV file containing a 'Class' column")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    _, metrics = train(df)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
