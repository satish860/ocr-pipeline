"""OCR Pipeline Benchmark Suite

Evaluates the QwenVL + Claude pipeline against the Omni OCR benchmark dataset.
"""

from .dataset import (
    load_benchmark_dataset,
    get_random_samples,
    get_samples_by_category,
    get_stratified_samples,
    get_dataset_stats,
    get_all_categories,
)

__all__ = [
    "load_benchmark_dataset",
    "get_random_samples",
    "get_samples_by_category",
    "get_stratified_samples",
    "get_dataset_stats",
    "get_all_categories",
]
