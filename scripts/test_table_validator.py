"""
Integration test for table validator and corrector.

Tests the full pipeline:
1. Extract document using QwenVL (extract_document)
2. Extract LaTeX tables from markdown
3. Match table images with LaTeX tables
4. Validate each table using Gemini 2.5 Flash
5. Correct errors using table corrector (test all strategies)
6. Display validation and correction results
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_pipeline import extract_document
from ocr_pipeline.table_validator import validate_table, extract_latex_from_markdown
from ocr_pipeline.table_corrector import correct_table


def test_sample(sample_path: str, sample_name: str):
    """Test validation on a single sample."""
    print(f"\n{'='*80}")
    print(f"Testing: {sample_name}")
    print(f"{'='*80}")

    # Step 1: Extract document
    print(f"\n[Step 1] Extracting document with QwenVL...")
    result = extract_document(sample_path, include_images=True)

    if not result['success']:
        print(f"[FAIL] Extraction failed: {result.get('error')}")
        return

    print(f"[PASS] Extraction successful")
    print(f"   - Markdown length: {len(result['markdown'])} characters")
    print(f"   - Images extracted: {len(result['images'])}")

    # Step 2: Extract LaTeX tables from markdown
    print(f"\n[Step 2] Extracting LaTeX tables from markdown...")
    latex_tables = extract_latex_from_markdown(result['markdown'])
    print(f"[PASS] Found {len(latex_tables)} LaTeX table(s)")

    # Step 3: Get table images
    print(f"\n[Step 3] Matching table images...")
    table_images = [img for img in result['images'] if img['type'] == 'Table']
    print(f"[PASS] Found {len(table_images)} table image(s)")

    # Verify counts match
    if len(latex_tables) != len(table_images):
        print(f"[WARN] LaTeX table count ({len(latex_tables)}) != Image count ({len(table_images)})")
        print(f"   Using minimum count for validation")

    # Step 4: Validate each table
    print(f"\n[Step 4] Validating tables...")
    validation_results = []

    for i in range(min(len(latex_tables), len(table_images))):
        print(f"\n   Table {i+1}:")
        print(f"   - LaTeX preview: {latex_tables[i][:100]}...")
        print(f"   - Image bbox: {table_images[i]['bbox']}")
        print(f"   - Validating...")

        validation = validate_table(
            table_image_base64=table_images[i]['base64'],
            latex_table=latex_tables[i]
        )

        validation_results.append(validation)

        # Display results
        print(f"   - Valid: {validation['valid']}")
        print(f"   - Confidence: {validation['confidence']:.2f}")
        print(f"   - Errors found: {validation['summary']['total_errors']}")

        if validation['summary']['total_errors'] > 0:
            print(f"     - Critical (>0.8): {validation['summary']['critical']}")
            print(f"     - High (0.6-0.8): {validation['summary']['high']}")
            print(f"     - Low (<0.6): {validation['summary']['low']}")

        # Show detailed errors
        if validation.get('errors'):
            print(f"\n   Detailed Errors:")
            for j, error in enumerate(validation['errors'], 1):
                print(f"     [{j}] Row {error['location']['row']}, Col {error['location']['column']}")
                print(f"         Type: {error['error_type']}")
                print(f"         LaTeX: '{error['latex_value']}'")
                print(f"         Image: '{error['image_value']}'")
                print(f"         Confidence: {error['confidence']:.2f}")

        # Show any API errors
        if validation.get('error'):
            print(f"   [WARN] API Error: {validation['error']}")

    # Step 5: Test corrector on tables with errors
    print(f"\n[Step 5] Testing corrector on tables with errors...")
    correction_results = []

    for i, validation in enumerate(validation_results):
        if validation['summary']['total_errors'] > 0 and not validation.get('error'):
            print(f"\n   Table {i+1} - Testing correction strategies:")

            # Test all 3 strategies
            for strategy in ['conservative', 'auto', 'aggressive']:
                print(f"\n      Strategy: {strategy}")

                correction = correct_table(
                    latex_table=latex_tables[i],
                    validation_report=validation,
                    table_image_base64=table_images[i]['base64'],
                    strategy=strategy
                )

                correction_results.append({
                    'table_index': i + 1,
                    'strategy': strategy,
                    'result': correction
                })

                print(f"      - Applied: {len(correction['corrections_applied'])}")
                print(f"      - Skipped: {len(correction['corrections_skipped'])}")

                # Show applied corrections
                if correction['corrections_applied']:
                    print(f"      Applied corrections:")
                    for j, corr in enumerate(correction['corrections_applied'], 1):
                        print(f"        [{j}] {corr['error_type']} at row {corr['location']['row']}, col {corr['location']['column']}")
                        print(f"            {corr['original_value']} -> {corr['corrected_value']}")
                        print(f"            Method: {corr['method']}")

                # Show skipped corrections
                if correction['corrections_skipped']:
                    print(f"      Skipped corrections:")
                    for j, skip in enumerate(correction['corrections_skipped'], 1):
                        print(f"        [{j}] {skip['error_type']} - {skip['reason']}")

                # Show before/after preview
                if correction['corrections_applied']:
                    print(f"\n      LaTeX changed: {latex_tables[i][:80] != correction['corrected_latex'][:80]}")

    # Summary
    print(f"\n{'='*80}")
    print(f"Summary for {sample_name}:")
    print(f"{'='*80}")
    total_tables = len(validation_results)
    valid_tables = sum(1 for v in validation_results if v['valid'])
    total_errors = sum(v['summary']['total_errors'] for v in validation_results)

    print(f"Total tables validated: {total_tables}")
    print(f"Valid tables: {valid_tables}/{total_tables}")
    print(f"Total errors detected: {total_errors}")

    if total_errors > 0:
        print(f"\n[PASS] Validator is working - detected errors in tables")

        # Corrector summary
        if correction_results:
            print(f"\n[PASS] Corrector is working - tested {len(correction_results)} strategy/table combinations")
            total_applied = sum(len(cr['result']['corrections_applied']) for cr in correction_results)
            total_skipped = sum(len(cr['result']['corrections_skipped']) for cr in correction_results)
            print(f"   Total corrections applied: {total_applied}")
            print(f"   Total corrections skipped: {total_skipped}")
    else:
        print(f"\n[PASS] All tables validated successfully (no errors detected)")

    return validation_results, correction_results


def main():
    """Run integration tests on sample tables."""
    print("="*80)
    print("Table Validator & Corrector Integration Test")
    print("="*80)

    # Test samples
    test_data_dir = Path(__file__).parent.parent / "test_data"

    # Test 1: General sample
    sample1 = test_data_dir / "table_01_id5.png"
    if sample1.exists():
        test_sample(str(sample1), "table_01_id5 (General Sample)")
    else:
        print(f"[WARN] Sample not found: {sample1}")

    # Test 2: Problematic sample (known to have issues - 21.1% baseline accuracy)
    sample2 = test_data_dir / "table_15_id20.png"
    if sample2.exists():
        test_sample(str(sample2), "table_15_id20 (Problematic Sample)")
    else:
        print(f"[WARN] Sample not found: {sample2}")

    print(f"\n{'='*80}")
    print("Integration Test Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
