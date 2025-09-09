# Fraud Detection Utilities

This repository provides simple utilities and a tiny training entrypoint for
experimentation with fraud detection models.

## Installation

```bash
pip install -e .[tests]
```

## Running tests

```bash
pytest -q
```

## Usage

Train a model by running the package as a module:

```bash
python -m fraud_detection --input data.csv --epochs 5 --lr 0.01 --outdir outputs
```

The same entrypoint is also available directly:

```bash
python -m fraud_detection.train --input data.csv --epochs 5 --lr 0.01 --outdir outputs
```

## Colab quick start

```python
!git clone https://github.com/your-username/fraud_detection.git
%cd fraud_detection
!pip install -e .[tests]
!python -m fraud_detection --input data.csv --outdir outputs
```
