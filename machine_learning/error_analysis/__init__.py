"""Public error-analysis API; implementation remains shared with post-match evaluation."""

from machine_learning.evaluation.model_error_analyzer import (
    build_error_rows,
    detect_failure_patterns,
)

__all__ = ["build_error_rows", "detect_failure_patterns"]
