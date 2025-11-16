"""
Debug script to see what validation/correction is actually doing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_pipeline import extract_document
from ocr_pipeline.table_validator import extract_latex_from_markdown, validate_table
from ocr_pipeline.table_corrector import correct_table


def debug_sample(image_path):
    """Debug validation on a single sample."""
    print(f"Processing: {image_path}")
    print("=" * 80)

    # Extract without validation
    print("\n[1] Extracting document...")
    result = extract_document(image_path, include_images=True, validate_tables=False)

    if not result['success']:
        print(f"ERROR: {result['error']}")
        return

    print(f"   Tables found: {len([img for img in result['images'] if img['type'] == 'Table'])}")

    # Extract LaTeX
    latex_tables = extract_latex_from_markdown(result['markdown'])
    table_images = [img for img in result['images'] if img['type'] == 'Table']

    print(f"   LaTeX tables extracted: {len(latex_tables)}")

    # Validate each table
    for i, (latex_table, table_image) in enumerate(zip(latex_tables, table_images), 1):
        print(f"\n[2.{i}] Validating table {i}...")
        print(f"   LaTeX preview: {latex_table[:150]}...")

        validation = validate_table(
            table_image_base64=table_image['base64'],
            latex_table=latex_table
        )

        print(f"   Valid: {validation['valid']}")
        print(f"   Errors found: {validation['summary']['total_errors']}")

        if validation['summary']['total_errors'] > 0:
            print(f"\n   Errors detected:")
            for j, error in enumerate(validation['errors'], 1):
                print(f"      [{j}] Type: {error['error_type']}")
                print(f"          Location: Row {error['location']['row']}, Col {error['location']['column']}")
                print(f"          LaTeX value: '{error['latex_value']}'")
                print(f"          Image value: '{error['image_value']}'")
                print(f"          Confidence: {error['confidence']:.2f}")

            # Test correction
            print(f"\n[3.{i}] Testing correction (conservative)...")
            correction = correct_table(
                latex_table=latex_table,
                validation_report=validation,
                table_image_base64=table_image['base64'],
                strategy="conservative"
            )

            print(f"   Corrections applied: {len(correction['corrections_applied'])}")
            print(f"   Corrections skipped: {len(correction['corrections_skipped'])}")

            if correction['corrections_applied']:
                print(f"\n   Applied corrections:")
                for j, corr in enumerate(correction['corrections_applied'], 1):
                    print(f"      [{j}] {corr['error_type']} at row {corr['location']['row']}, col {corr['location']['column']}")
                    print(f"          '{corr['original_value']}' -> '{corr['corrected_value']}'")
                    print(f"          Method: {corr['method']}")

            if correction['corrections_skipped']:
                print(f"\n   Skipped corrections:")
                for j, skip in enumerate(correction['corrections_skipped'], 1):
                    print(f"      [{j}] {skip['error_type']} - {skip['reason']}")

            # Show before/after LaTeX
            print(f"\n   LaTeX changed: {latex_table != correction['corrected_latex']}")
            if latex_table != correction['corrected_latex']:
                print(f"\n   BEFORE (first 200 chars):")
                print(f"   {latex_table[:200]}")
                print(f"\n   AFTER (first 200 chars):")
                print(f"   {correction['corrected_latex'][:200]}")
        else:
            print(f"   No errors detected - table is valid!")


if __name__ == "__main__":
    # Debug the worst samples
    test_data = Path(__file__).parent.parent / "test_data"

    print("=" * 80)
    print("VALIDATION DEBUG - Sample ID=4")
    print("=" * 80)
    debug_sample(str(test_data / "table_06_id4.png"))

    print("\n\n")
    print("=" * 80)
    print("VALIDATION DEBUG - Sample ID=20")
    print("=" * 80)
    debug_sample(str(test_data / "table_15_id20.png"))
