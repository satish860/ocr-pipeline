"""
Test only the worst performing samples to quickly verify improvements.
"""

import sys
import time
import json
import tempfile
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from src.ocr_pipeline.qwen_extractor import extract_document
from benchmark.claude_extractor import extract_json_from_markdown
from benchmark.evaluator import compare_json


def load_sample(sample_id):
    """Load a specific sample by ID from test_data/"""
    test_data_dir = Path(__file__).parent.parent / "test_data"

    # Find the file for this sample ID
    json_files = list(test_data_dir.glob(f"*_id{sample_id}.json"))

    if not json_files:
        raise FileNotFoundError(f"No sample found with ID {sample_id}")

    json_file = json_files[0]
    png_file = json_file.with_suffix('.png')

    # Load JSON metadata
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Load image
    image = Image.open(png_file)

    return {
        'id': data['id'],
        'schema': data['schema'],
        'json_gt': data['json_gt'],
        'metadata': data['metadata'],
        'image': image
    }


def test_with_validation(sample_id, sample, temp_path, validation_enabled=True, strategy="conservative"):
    """Test a sample with optional validation."""
    # Step 1: QwenVL extraction with validation
    mode_desc = f"validation ({strategy})" if validation_enabled else "NO validation"
    print(f"\n[1/3] QwenVL + {mode_desc}...", end=" ", flush=True)
    start_time = time.time()
    qwen_result = extract_document(
        temp_path,
        include_images=True,
        include_usage=True,
        convert_tables_to_html=False,  # Disable HTML conversion
        validate_tables=validation_enabled,  # Enable/disable validation
        correction_strategy=strategy
    )
    qwen_time = time.time() - start_time

    if not qwen_result['success']:
        print(f"FAILED - {qwen_result['error']}")
        return None

    print(f"OK ({qwen_time:.1f}s)")

    # Step 2: Claude JSON extraction
    print(f"[2/3] Claude JSON extraction...", end=" ", flush=True)
    start_time = time.time()
    claude_result = extract_json_from_markdown(
        markdown=qwen_result['markdown'],
        json_schema=sample['schema'],
        extracted_images=qwen_result['images'],
        include_usage=True
    )
    claude_time = time.time() - start_time

    if not claude_result['success']:
        print(f"FAILED - {claude_result['error']}")
        return None

    print(f"OK ({claude_time:.1f}s)")

    # Step 3: Evaluation
    print(f"[3/3] Evaluating...", end=" ", flush=True)
    evaluation = compare_json(claude_result['json'], sample['json_gt'])

    total_fields = evaluation['total_fields']
    diff_fields = evaluation['different_fields']
    accuracy = evaluation['accuracy'] * 100

    print(f"Accuracy: {accuracy:.1f}%")

    # Show top errors
    if diff_fields > 0 and len(evaluation['differences']) > 0:
        print(f"\nTop 5 errors:")
        for j, diff in enumerate(evaluation['differences'][:5], 1):
            print(f"  {j}. {diff['path']}")
            print(f"     Predicted: {diff['predicted']}")
            print(f"     Expected:  {diff['ground_truth']}")
    else:
        print("\n[PASS] PERFECT! No errors!")

    return {
        'accuracy': accuracy,
        'qwen_time': qwen_time,
        'claude_time': claude_time,
        'total_fields': total_fields,
        'diff_fields': diff_fields,
        'evaluation': evaluation
    }


def main():
    # Test the 2 worst samples
    WORST_SAMPLES = [4, 20]  # ID 4 (3.8% with HTML) and ID 20 (21.1% baseline)

    print("=" * 70)
    print("TESTING WORST SAMPLES - BASELINE vs VALIDATION")
    print("=" * 70)

    for sample_id in WORST_SAMPLES:
        print(f"\n{'='*70}")
        print(f"Sample ID={sample_id}")
        print(f"{'='*70}")

        # Load sample
        sample = load_sample(sample_id)

        # Create temp file for image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            sample['image'].save(tmp.name, format='PNG')
            temp_path = tmp.name

        try:
            # Test 1: Baseline (no validation)
            print(f"\n--- BASELINE (No Validation) ---")
            baseline_result = test_with_validation(
                sample_id, sample, temp_path,
                validation_enabled=False
            )

            # Test 2: With validation (conservative strategy)
            print(f"\n--- WITH VALIDATION (Conservative) ---")
            validation_result = test_with_validation(
                sample_id, sample, temp_path,
                validation_enabled=True,
                strategy="conservative"
            )

            # Comparison
            if baseline_result and validation_result:
                print(f"\n{'='*70}")
                print(f"COMPARISON - Sample ID={sample_id}")
                print(f"{'='*70}")
                print(f"Baseline accuracy:    {baseline_result['accuracy']:.1f}%")
                print(f"Validation accuracy:  {validation_result['accuracy']:.1f}%")
                improvement = validation_result['accuracy'] - baseline_result['accuracy']
                print(f"Improvement:          {improvement:+.1f}%")
                print(f"")
                print(f"Baseline errors:      {baseline_result['diff_fields']}/{baseline_result['total_fields']}")
                print(f"Validation errors:    {validation_result['diff_fields']}/{validation_result['total_fields']}")
                errors_fixed = baseline_result['diff_fields'] - validation_result['diff_fields']
                print(f"Errors fixed:         {errors_fixed}")

                if improvement > 0:
                    print(f"\n[PASS] Validation IMPROVED accuracy!")
                elif improvement == 0:
                    print(f"\n[INFO] No change in accuracy")
                else:
                    print(f"\n[WARN] Validation DECREASED accuracy")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
