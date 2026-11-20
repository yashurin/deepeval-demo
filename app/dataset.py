"""Backward-compatible import path for the sample questions.

The questions live in ``app.sample_data``. This module re-exports them so
``from app.dataset import GOLDENS`` keeps working.
"""

from app.sample_data import GOLDENS, Golden, case_input

__all__ = ["GOLDENS", "Golden", "case_input"]
