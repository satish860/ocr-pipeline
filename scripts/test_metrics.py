"""
Metrics Prototype for OCR Benchmark Evaluation

This script implements and tests the core evaluation metrics:
1. Text Similarity (Levenshtein distance)
2. JSON Accuracy (field-level comparison)

Usage: uv run python scripts/test_metrics.py
"""

from Levenshtein import distance
from typing import Any, Dict
import json


def calculate_text_similarity(predicted: str, ground_truth: str) -> float:
    """
    Calculate normalized Levenshtein distance between markdown outputs.

    Args:
        predicted: Pipeline's markdown output
        ground_truth: Ground truth markdown from dataset

    Returns:
        Score from 0.0 to 1.0 (1.0 = perfect match)
    """
    edit_dist = distance(predicted, ground_truth)
    max_len = max(len(predicted), len(ground_truth))

    if max_len == 0:
        return 1.0

    similarity = 1.0 - (edit_dist / max_len)
    return similarity


def count_fields(obj: Any) -> int:
    """
    Recursively count all fields in nested JSON structure.

    Args:
        obj: JSON object (dict, list, or primitive)

    Returns:
        Total number of fields
    """
    if isinstance(obj, dict):
        return sum(1 + count_fields(v) for v in obj.values())
    elif isinstance(obj, list):
        return sum(count_fields(item) for item in obj)
    else:
        return 1


def count_differences(pred: Any, truth: Any, path: str = "root") -> tuple[int, list]:
    """
    Recursively count differing fields between predicted and ground truth JSON.

    Args:
        pred: Predicted JSON object
        truth: Ground truth JSON object
        path: Current path in JSON structure (for debugging)

    Returns:
        Tuple of (difference_count, list of difference descriptions)
    """
    differences = []
    diff_count = 0

    # Type mismatch
    if type(pred) != type(truth):
        differences.append(f"{path}: Type mismatch ({type(pred).__name__} vs {type(truth).__name__})")
        return count_fields(truth), differences

    # Compare dictionaries
    if isinstance(truth, dict):
        all_keys = set(truth.keys()) | set(pred.keys())

        for key in all_keys:
            key_path = f"{path}.{key}"

            # Missing key in prediction
            if key not in pred:
                missing_fields = count_fields(truth[key])
                diff_count += missing_fields
                differences.append(f"{key_path}: Missing in prediction ({missing_fields} fields)")

            # Extra key in prediction
            elif key not in truth:
                extra_fields = count_fields(pred[key])
                diff_count += extra_fields
                differences.append(f"{key_path}: Extra in prediction ({extra_fields} fields)")

            # Recursively compare nested structures
            else:
                nested_diff_count, nested_differences = count_differences(
                    pred[key], truth[key], key_path
                )
                diff_count += nested_diff_count
                differences.extend(nested_differences)

    # Compare lists
    elif isinstance(truth, list):
        if len(pred) != len(truth):
            differences.append(f"{path}: Array length mismatch ({len(pred)} vs {len(truth)})")
            diff_count += abs(len(pred) - len(truth))

        # Compare elements up to the shorter length
        for i in range(min(len(pred), len(truth))):
            nested_diff_count, nested_differences = count_differences(
                pred[i], truth[i], f"{path}[{i}]"
            )
            diff_count += nested_diff_count
            differences.extend(nested_differences)

        # Count extra/missing elements
        if len(pred) < len(truth):
            for i in range(len(pred), len(truth)):
                missing_fields = count_fields(truth[i])
                diff_count += missing_fields
                differences.append(f"{path}[{i}]: Missing element ({missing_fields} fields)")

        elif len(pred) > len(truth):
            for i in range(len(truth), len(pred)):
                extra_fields = count_fields(pred[i])
                diff_count += extra_fields
                differences.append(f"{path}[{i}]: Extra element ({extra_fields} fields)")

    # Compare primitives
    else:
        if pred != truth:
            diff_count = 1
            differences.append(f"{path}: Value mismatch ({repr(pred)} vs {repr(truth)})")

    return diff_count, differences


def calculate_json_accuracy(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> tuple[float, dict]:
    """
    Calculate field-level accuracy for structured JSON extraction.

    Args:
        predicted: Pipeline's JSON output
        ground_truth: Ground truth JSON from dataset

    Returns:
        Tuple of (accuracy_score, detailed_metrics)
        - accuracy_score: Float from 0.0 to 1.0 (1.0 = perfect match)
        - detailed_metrics: Dict with total_fields, diff_fields, and differences list
    """
    total_fields = count_fields(ground_truth)
    diff_count, differences = count_differences(predicted, ground_truth)

    if total_fields == 0:
        accuracy = 1.0
    else:
        accuracy = 1.0 - (diff_count / total_fields)

    detailed_metrics = {
        "total_fields": total_fields,
        "diff_fields": diff_count,
        "matching_fields": total_fields - diff_count,
        "accuracy": accuracy,
        "differences": differences
    }

    return accuracy, detailed_metrics


def test_text_similarity():
    """Test text similarity metric with examples."""
    print("Testing Text Similarity Metric")
    print("=" * 60)

    # Test 1: Perfect match
    text1 = "Hello world"
    text2 = "Hello world"
    score = calculate_text_similarity(text1, text2)
    print(f"\nTest 1: Perfect match")
    print(f"  Text 1: {repr(text1)}")
    print(f"  Text 2: {repr(text2)}")
    print(f"  Similarity: {score:.4f} (expected: 1.0000)")

    # Test 2: Partial match
    text1 = "The quick brown fox"
    text2 = "The quick brown dog"
    score = calculate_text_similarity(text1, text2)
    print(f"\nTest 2: Partial match")
    print(f"  Text 1: {repr(text1)}")
    print(f"  Text 2: {repr(text2)}")
    print(f"  Similarity: {score:.4f}")

    # Test 3: No match
    text1 = "Hello"
    text2 = "World"
    score = calculate_text_similarity(text1, text2)
    print(f"\nTest 3: No match")
    print(f"  Text 1: {repr(text1)}")
    print(f"  Text 2: {repr(text2)}")
    print(f"  Similarity: {score:.4f}")

    # Test 4: Empty strings
    text1 = ""
    text2 = ""
    score = calculate_text_similarity(text1, text2)
    print(f"\nTest 4: Empty strings")
    print(f"  Similarity: {score:.4f} (expected: 1.0000)")

    print("\n" + "=" * 60)


def test_json_accuracy():
    """Test JSON accuracy metric with examples."""
    print("\n\nTesting JSON Accuracy Metric")
    print("=" * 60)

    # Test 1: Perfect match
    pred = {"name": "John", "age": 30}
    truth = {"name": "John", "age": 30}
    score, metrics = calculate_json_accuracy(pred, truth)
    print(f"\nTest 1: Perfect match")
    print(f"  Predicted: {pred}")
    print(f"  Truth: {truth}")
    print(f"  Accuracy: {score:.4f} (expected: 1.0000)")
    print(f"  Total fields: {metrics['total_fields']}, Differences: {metrics['diff_fields']}")

    # Test 2: Partial match
    pred = {"name": "John", "age": 25}
    truth = {"name": "John", "age": 30}
    score, metrics = calculate_json_accuracy(pred, truth)
    print(f"\nTest 2: Partial match (value difference)")
    print(f"  Predicted: {pred}")
    print(f"  Truth: {truth}")
    print(f"  Accuracy: {score:.4f}")
    print(f"  Total fields: {metrics['total_fields']}, Differences: {metrics['diff_fields']}")
    print(f"  Differences: {metrics['differences']}")

    # Test 3: Missing field
    pred = {"name": "John"}
    truth = {"name": "John", "age": 30}
    score, metrics = calculate_json_accuracy(pred, truth)
    print(f"\nTest 3: Missing field")
    print(f"  Predicted: {pred}")
    print(f"  Truth: {truth}")
    print(f"  Accuracy: {score:.4f}")
    print(f"  Total fields: {metrics['total_fields']}, Differences: {metrics['diff_fields']}")
    print(f"  Differences: {metrics['differences']}")

    # Test 4: Nested structure
    pred = {"person": {"name": "John", "age": 30}, "city": "NYC"}
    truth = {"person": {"name": "John", "age": 30}, "city": "NYC"}
    score, metrics = calculate_json_accuracy(pred, truth)
    print(f"\nTest 4: Nested structure (perfect match)")
    print(f"  Predicted: {pred}")
    print(f"  Truth: {truth}")
    print(f"  Accuracy: {score:.4f} (expected: 1.0000)")
    print(f"  Total fields: {metrics['total_fields']}, Differences: {metrics['diff_fields']}")

    # Test 5: Array comparison
    pred = {"scores": [90, 85, 88]}
    truth = {"scores": [90, 85, 88]}
    score, metrics = calculate_json_accuracy(pred, truth)
    print(f"\nTest 5: Array (perfect match)")
    print(f"  Predicted: {pred}")
    print(f"  Truth: {truth}")
    print(f"  Accuracy: {score:.4f} (expected: 1.0000)")
    print(f"  Total fields: {metrics['total_fields']}, Differences: {metrics['diff_fields']}")

    # Test 6: Array length mismatch
    pred = {"scores": [90, 85]}
    truth = {"scores": [90, 85, 88]}
    score, metrics = calculate_json_accuracy(pred, truth)
    print(f"\nTest 6: Array length mismatch")
    print(f"  Predicted: {pred}")
    print(f"  Truth: {truth}")
    print(f"  Accuracy: {score:.4f}")
    print(f"  Total fields: {metrics['total_fields']}, Differences: {metrics['diff_fields']}")
    print(f"  Differences: {metrics['differences']}")

    print("\n" + "=" * 60)


def test_with_real_sample():
    """Test metrics with real benchmark sample."""
    print("\n\nTesting with Real Benchmark Sample")
    print("=" * 60)

    from pathlib import Path

    # Check if sample files exist
    sample_dir = Path("input/benchmark_samples")
    if not sample_dir.exists():
        print("Benchmark samples not found. Run explore_dataset.py first.")
        return

    # Test with Sample 1 (TABLE)
    markdown_path = sample_dir / "sample_1_TABLE_truth.md"
    json_path = sample_dir / "sample_1_TABLE_truth.json"

    if markdown_path.exists() and json_path.exists():
        print("\nSample 1 (TABLE):")
        print("-" * 60)

        # Load ground truth
        truth_markdown = markdown_path.read_text(encoding="utf-8")
        truth_json = json.loads(json_path.read_text(encoding="utf-8"))

        print(f"Ground truth markdown length: {len(truth_markdown)} chars")
        print(f"Ground truth JSON keys: {list(truth_json.keys())}")
        print(f"Ground truth JSON structure: {json.dumps(truth_json, indent=2)[:200]}...")

        # Simulate predicted output (for testing, use slightly modified versions)
        print("\n\nSimulating predictions (with slight errors):")

        # Test 1: Perfect match
        pred_markdown = truth_markdown
        score = calculate_text_similarity(pred_markdown, truth_markdown)
        print(f"\nMarkdown similarity (perfect match): {score:.4f}")

        # Test 2: 95% match (remove a few characters)
        pred_markdown = truth_markdown[:int(len(truth_markdown) * 0.95)]
        score = calculate_text_similarity(pred_markdown, truth_markdown)
        print(f"Markdown similarity (95%% length): {score:.4f}")

        # Test 3: JSON accuracy (perfect match)
        pred_json = truth_json
        score, metrics = calculate_json_accuracy(pred_json, truth_json)
        print(f"\nJSON accuracy (perfect match): {score:.4f}")
        print(f"Total fields: {metrics['total_fields']}")

    print("\n" + "=" * 60)


def main():
    """Run all metric tests."""
    test_text_similarity()
    test_json_accuracy()
    test_with_real_sample()
    print("\n\nAll tests complete!")


if __name__ == "__main__":
    main()
