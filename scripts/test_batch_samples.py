"""
Batch Test Script for Multiple Benchmark Samples

Tests JSON extraction on multiple samples and generates summary report.
Usage: uv run python scripts/test_batch_samples.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_pipeline.json_extractor import JSONExtractor


def count_fields(obj: Any) -> int:
    """Recursively count all fields in nested JSON structure."""
    if isinstance(obj, dict):
        return sum(1 + count_fields(v) for v in obj.values())
    elif isinstance(obj, list):
        return sum(count_fields(item) for item in obj)
    else:
        return 1


def count_differences(pred: Any, truth: Any, path: str = "root") -> tuple[int, list]:
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
                differences.append(f"{key_path}: Missing ({missing_fields} fields)")
            elif key not in truth:
                extra_fields = count_fields(pred[key])
                diff_count += extra_fields
                differences.append(f"{key_path}: Extra ({extra_fields} fields)")
            else:
                nested_diff_count, nested_differences = count_differences(
                    pred[key], truth[key], key_path
                )
                diff_count += nested_diff_count
                differences.extend(nested_differences)

    elif isinstance(truth, list):
        if len(pred) != len(truth):
            diff_count += abs(len(pred) - len(truth))
        for i in range(min(len(pred), len(truth))):
            nested_diff_count, nested_differences = count_differences(
                pred[i], truth[i], f"{path}[{i}]"
            )
            diff_count += nested_diff_count
            differences.extend(nested_differences)

    else:
        if pred != truth:
            diff_count = 1
            differences.append(f"{path}: Value mismatch")

    return diff_count, differences


def calculate_json_accuracy(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> tuple[float, dict]:
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
        "differences": differences[:5]  # Only keep first 5 for reporting
    }


def test_sample(sample_name: str, extractor: JSONExtractor) -> Dict[str, Any]:
    """Test a single sample and return results."""
    print(f"\nTesting: {sample_name}")
    print("-" * 60)

    sample_dir = Path("input/benchmark_samples")
    output_dir = Path("output")

    markdown_path = output_dir / f"{sample_name}_complete.md"
    schema_path = sample_dir / f"{sample_name}_schema.json"
    truth_json_path = sample_dir / f"{sample_name}_truth.json"

    # Check files exist
    if not markdown_path.exists():
        print(f"  ERROR: Missing markdown file")
        return {"sample": sample_name, "error": "Missing markdown", "accuracy": 0.0}

    if not schema_path.exists():
        print(f"  ERROR: Missing schema file")
        return {"sample": sample_name, "error": "Missing schema", "accuracy": 0.0}

    if not truth_json_path.exists():
        print(f"  ERROR: Missing ground truth")
        return {"sample": sample_name, "error": "Missing truth", "accuracy": 0.0}

    # Load files
    try:
        markdown = markdown_path.read_text(encoding="utf-8")
        json_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        ground_truth = json.loads(truth_json_path.read_text(encoding="utf-8"))

        # Extract JSON (with multimodal support)
        predicted_json = extractor.extract(markdown, json_schema, image_dir=str(output_dir))

        # Save predicted JSON
        output_json_path = output_dir / f"{sample_name}_extracted.json"
        output_json_path.write_text(json.dumps(predicted_json, indent=2), encoding="utf-8")

        # Calculate accuracy
        accuracy, metrics = calculate_json_accuracy(predicted_json, ground_truth)

        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Fields: {metrics['total_fields']} total, {metrics['diff_fields']} differences")

        if metrics['differences']:
            print(f"  Top differences:")
            for diff in metrics['differences'][:3]:
                print(f"    - {diff}")

        return {
            "sample": sample_name,
            "accuracy": accuracy,
            "total_fields": metrics['total_fields'],
            "diff_fields": metrics['diff_fields'],
            "success": True
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        return {"sample": sample_name, "error": str(e), "accuracy": 0.0, "success": False}


def main():
    """Run batch testing on all samples."""
    print("=" * 60)
    print("BATCH BENCHMARK TEST")
    print("=" * 60)

    samples = [
        "sample_1_TABLE",
        "sample_2_CHART",
        "sample_3_CHART",
        "sample_4_TABLE",
        "sample_5_TABLE",
        "sample_6_TABLE"
    ]

    print(f"\nTesting {len(samples)} samples...")
    print(f"Initializing JSON extractor (Claude Haiku 3.5)...")

    extractor = JSONExtractor()
    results = []

    # Test each sample
    for sample in samples:
        result = test_sample(sample, extractor)
        results.append(result)

    # Generate summary report
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]

    if successful_results:
        avg_accuracy = sum(r["accuracy"] for r in successful_results) / len(successful_results)
        total_fields = sum(r["total_fields"] for r in successful_results)
        total_diffs = sum(r["diff_fields"] for r in successful_results)

        print(f"\nSuccessful Tests: {len(successful_results)}/{len(results)}")
        print(f"Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        print(f"Total Fields Evaluated: {total_fields}")
        print(f"Total Field Differences: {total_diffs}")

        print("\nPer-Sample Results:")
        print(f"{'Sample':<25} {'Accuracy':<12} {'Fields':<10} {'Diffs'}")
        print("-" * 60)
        for r in successful_results:
            print(f"{r['sample']:<25} {r['accuracy']:.4f} ({r['accuracy']*100:5.2f}%)  {r['total_fields']:<10} {r['diff_fields']}")

    if failed_results:
        print(f"\nFailed Tests: {len(failed_results)}")
        for r in failed_results:
            print(f"  - {r['sample']}: {r.get('error', 'Unknown error')}")

    # Save results to JSON
    results_path = Path("output/batch_test_results.json")
    results_path.write_text(json.dumps({
        "summary": {
            "total_samples": len(results),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "average_accuracy": avg_accuracy if successful_results else 0.0,
            "total_fields": total_fields if successful_results else 0,
            "total_differences": total_diffs if successful_results else 0
        },
        "results": results
    }, indent=2), encoding="utf-8")

    print(f"\nDetailed results saved to: {results_path}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
