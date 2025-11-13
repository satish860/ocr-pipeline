"""
Pipeline Validation Script

Compares OCR pipeline output with benchmark ground truth.
Usage: uv run python scripts/validate_pipeline.py
"""

import json
from pathlib import Path
from Levenshtein import distance


def calculate_text_similarity(predicted: str, ground_truth: str) -> float:
    """Calculate normalized Levenshtein distance."""
    edit_dist = distance(predicted, ground_truth)
    max_len = max(len(predicted), len(ground_truth))

    if max_len == 0:
        return 1.0

    return 1.0 - (edit_dist / max_len)


def clean_markdown(text: str) -> str:
    """
    Clean markdown for comparison.
    Remove metadata comments and normalize whitespace.
    """
    lines = []
    for line in text.split('\n'):
        # Skip metadata comments
        if line.strip().startswith('<!--') and line.strip().endswith('-->'):
            continue
        lines.append(line)

    return '\n'.join(lines).strip()


def validate_sample(sample_name: str):
    """
    Validate pipeline output against ground truth for a sample.

    Args:
        sample_name: Sample identifier (e.g., "sample_1_TABLE")
    """
    print(f"\nValidating: {sample_name}")
    print("=" * 60)

    # Paths
    pipeline_output = Path("output") / f"{sample_name}_complete.md"
    ground_truth_md = Path("input/benchmark_samples") / f"{sample_name}_truth.md"

    # Check files exist
    if not pipeline_output.exists():
        print(f"Error: Pipeline output not found: {pipeline_output}")
        return

    if not ground_truth_md.exists():
        print(f"Error: Ground truth not found: {ground_truth_md}")
        return

    # Load files
    pipeline_text = pipeline_output.read_text(encoding="utf-8")
    ground_truth_text = ground_truth_md.read_text(encoding="utf-8")

    # Handle JSON string in ground truth (if wrapped in quotes)
    if ground_truth_text.startswith('"') and ground_truth_text.endswith('"'):
        ground_truth_text = json.loads(ground_truth_text)

    # Clean both texts
    pipeline_clean = clean_markdown(pipeline_text)
    ground_truth_clean = clean_markdown(ground_truth_text)

    print(f"\nPipeline output length: {len(pipeline_clean)} chars")
    print(f"Ground truth length: {len(ground_truth_clean)} chars")

    # Calculate similarity
    similarity = calculate_text_similarity(pipeline_clean, ground_truth_clean)

    print(f"\nText Similarity Score: {similarity:.4f} ({similarity*100:.2f}%)")

    # Show sample comparison
    print("\n" + "-" * 60)
    print("Pipeline output (first 200 chars):")
    print(pipeline_clean[:200])
    print("\n" + "-" * 60)
    print("Ground truth (first 200 chars):")
    print(ground_truth_clean[:200])
    print("-" * 60)

    # Identify key differences
    print("\nKey observations:")
    if similarity >= 0.95:
        print("  - Excellent match! Very minor differences.")
    elif similarity >= 0.85:
        print("  - Good match with some formatting differences.")
    elif similarity >= 0.70:
        print("  - Moderate match. Notable differences in structure or content.")
    else:
        print("  - Low match. Significant differences detected.")

    # Character-level analysis
    print(f"\nCharacter difference: {abs(len(pipeline_clean) - len(ground_truth_clean))} chars")

    if similarity < 1.0:
        print("\nPotential issues:")
        # Check for bold formatting
        if "**" in ground_truth_clean and "**" not in pipeline_clean:
            print("  - Missing bold formatting (**text**)")

        # Check for extra whitespace
        if len(pipeline_clean) > len(ground_truth_clean):
            print("  - Pipeline output has extra characters")
        elif len(pipeline_clean) < len(ground_truth_clean):
            print("  - Pipeline output is missing some content")

    return similarity


def main():
    """Validate all available samples."""
    print("OCR Pipeline Validation")
    print("=" * 60)

    samples = [
        "sample_1_TABLE",
        "sample_2_CHART"
    ]

    scores = []
    for sample in samples:
        score = validate_sample(sample)
        if score is not None:
            scores.append(score)

    if scores:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Samples tested: {len(scores)}")
        print(f"Average similarity: {sum(scores)/len(scores):.4f} ({sum(scores)/len(scores)*100:.2f}%)")
        print(f"Best score: {max(scores):.4f}")
        print(f"Worst score: {min(scores):.4f}")


if __name__ == "__main__":
    main()
