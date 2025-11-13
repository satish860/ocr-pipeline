"""
Test JSON Extraction on Benchmark Sample

This script tests the JSON extractor on real benchmark data.
Usage: uv run python scripts/test_json_extraction.py [sample_name]
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_pipeline.json_extractor import JSONExtractor


def count_fields(obj) -> int:
    """Recursively count all fields in nested JSON structure."""
    if isinstance(obj, dict):
        return sum(1 + count_fields(v) for v in obj.values())
    elif isinstance(obj, list):
        return sum(count_fields(item) for item in obj)
    else:
        return 1


def count_differences(pred, truth, path: str = "root") -> tuple[int, list]:
    """Recursively count differing fields."""
    differences = []
    diff_count = 0

    if type(pred) != type(truth):
        differences.append(f"{path}: Type mismatch ({type(pred).__name__} vs {type(truth).__name__})")
        return count_fields(truth), differences

    if isinstance(truth, dict):
        all_keys = set(truth.keys()) | set(pred.keys())
        for key in all_keys:
            key_path = f"{path}.{key}"
            if key not in pred:
                missing_fields = count_fields(truth[key])
                diff_count += missing_fields
                differences.append(f"{key_path}: Missing in prediction ({missing_fields} fields)")
            elif key not in truth:
                extra_fields = count_fields(pred[key])
                diff_count += extra_fields
                differences.append(f"{key_path}: Extra in prediction ({extra_fields} fields)")
            else:
                nested_diff_count, nested_differences = count_differences(
                    pred[key], truth[key], key_path
                )
                diff_count += nested_diff_count
                differences.extend(nested_differences)

    elif isinstance(truth, list):
        if len(pred) != len(truth):
            differences.append(f"{path}: Array length mismatch ({len(pred)} vs {len(truth)})")
            diff_count += abs(len(pred) - len(truth))

        for i in range(min(len(pred), len(truth))):
            nested_diff_count, nested_differences = count_differences(
                pred[i], truth[i], f"{path}[{i}]"
            )
            diff_count += nested_diff_count
            differences.extend(nested_differences)

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

    else:
        if pred != truth:
            diff_count = 1
            differences.append(f"{path}: Value mismatch ({repr(pred)} vs {repr(truth)})")

    return diff_count, differences


def calculate_json_accuracy(predicted: dict, ground_truth: dict) -> tuple[float, dict]:
    """Calculate JSON accuracy score."""
    total_fields = count_fields(ground_truth)
    diff_count, differences = count_differences(predicted, ground_truth)

    if total_fields == 0:
        accuracy = 1.0
    else:
        accuracy = 1.0 - (diff_count / total_fields)

    return accuracy, {
        "total_fields": total_fields,
        "diff_fields": diff_count,
        "accuracy": accuracy,
        "differences": differences
    }


def test_json_extraction(sample_name: str = "sample_1_TABLE"):
    """Test JSON extraction on a benchmark sample."""
    print(f"\nTesting JSON Extraction: {sample_name}")
    print("=" * 60)

    # Paths
    sample_dir = Path("input/benchmark_samples")
    output_dir = Path("output")

    markdown_path = output_dir / f"{sample_name}_complete.md"
    schema_path = sample_dir / f"{sample_name}_schema.json"
    truth_json_path = sample_dir / f"{sample_name}_truth.json"

    # Check files exist
    if not markdown_path.exists():
        print(f"Error: Pipeline markdown not found: {markdown_path}")
        print("Run the pipeline first: uv run python test_layout_detector.py input/benchmark_samples/{sample_name}.png")
        return None

    if not schema_path.exists():
        print(f"Error: JSON schema not found: {schema_path}")
        return None

    if not truth_json_path.exists():
        print(f"Error: Ground truth JSON not found: {truth_json_path}")
        return None

    # Load files
    print("\nLoading files...")
    markdown = markdown_path.read_text(encoding="utf-8")
    json_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(truth_json_path.read_text(encoding="utf-8"))

    print(f"  Markdown length: {len(markdown)} chars")
    print(f"  JSON schema keys: {list(json_schema.keys())}")
    print(f"  Ground truth keys: {list(ground_truth.keys())}")

    # Initialize extractor
    print("\nInitializing JSON extractor (Claude Haiku 3.5)...")
    extractor = JSONExtractor()

    # Extract JSON
    print("Extracting JSON from markdown...")
    try:
        predicted_json = extractor.extract(markdown, json_schema)
        print("  Extraction successful!")
        print(f"  Predicted JSON keys: {list(predicted_json.keys())}")

        # Save predicted JSON
        output_json_path = output_dir / f"{sample_name}_extracted.json"
        output_json_path.write_text(json.dumps(predicted_json, indent=2), encoding="utf-8")
        print(f"  Saved to: {output_json_path}")

    except Exception as e:
        print(f"  Extraction failed: {e}")
        return None

    # Calculate accuracy
    print("\n" + "-" * 60)
    print("Evaluating JSON Accuracy")
    print("-" * 60)

    accuracy, metrics = calculate_json_accuracy(predicted_json, ground_truth)

    print(f"\nJSON Accuracy Score: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total fields: {metrics['total_fields']}")
    print(f"Matching fields: {metrics['total_fields'] - metrics['diff_fields']}")
    print(f"Different fields: {metrics['diff_fields']}")

    if metrics['differences']:
        print("\nDifferences found:")
        for diff in metrics['differences'][:10]:  # Show first 10
            print(f"  - {diff}")
        if len(metrics['differences']) > 10:
            print(f"  ... and {len(metrics['differences']) - 10} more differences")

    # Show sample data
    print("\n" + "-" * 60)
    print("Sample Comparison")
    print("-" * 60)
    print("\nPredicted JSON (first item):")
    print(json.dumps(predicted_json, indent=2)[:300])
    print("\nGround Truth JSON (first item):")
    print(json.dumps(ground_truth, indent=2)[:300])

    return accuracy


def main():
    """Run JSON extraction test."""
    sample_name = sys.argv[1] if len(sys.argv) > 1 else "sample_1_TABLE"

    score = test_json_extraction(sample_name)

    if score is not None:
        print("\n" + "=" * 60)
        print("Test Complete")
        print("=" * 60)
        print(f"Final JSON Accuracy: {score:.4f} ({score*100:.2f}%)")

        if score >= 0.90:
            print("Result: Excellent!")
        elif score >= 0.70:
            print("Result: Good - some improvements needed")
        elif score >= 0.50:
            print("Result: Moderate - significant improvements needed")
        else:
            print("Result: Poor - major issues detected")


if __name__ == "__main__":
    main()
