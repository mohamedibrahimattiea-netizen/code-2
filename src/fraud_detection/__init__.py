"""Simple utilities for fraud detection analysis."""
from .train import train


def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

__all__ = ["add", "train"]
