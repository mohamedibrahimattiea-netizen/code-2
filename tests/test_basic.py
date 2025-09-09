import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fraud_detection import add

def test_add():
    assert add(1, 2) == 3
