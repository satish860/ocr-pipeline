"""
Batch process PDF: Convert to images, run OCR pipeline on all pages.

Usage: uv run python scripts/batch_process_pdf.py <path_to_pdf> [--keep-images]
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.pdf_to_images import pdf_to_images
from test_layout_detector import test_ocr_pipeline


def batch_process_pdf(pdf_path: str, cleanup_images: bool = True):
    """
    Process all pages of a PDF through the OCR pipeline.

    Args:
        pdf_path: Path to PDF file
        cleanup_images: Delete page images after processing (default: True)
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        return

    print("="*70)
    print("BATCH PDF PROCESSING")
    print("="*70)
    print(f"\nPDF: {pdf_path.name}")
    print(f"Cleanup images after processing: {cleanup_images}\n")

    # Step 1: Convert PDF to images
    print("\n[STEP 1] Converting PDF to images...")
    try:
        image_paths = pdf_to_images(str(pdf_path))
        print(f"\n[OK] Created {len(image_paths)} page images")
    except Exception as e:
        print(f"\n[ERROR] Failed to convert PDF: {e}")
        return

    # Step 2: Process each page
    print(f"\n[STEP 2] Processing {len(image_paths)} pages through OCR pipeline...")
    successful = 0
    failed = 0

    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n{'='*70}")
        print(f"PAGE {idx}/{len(image_paths)}: {image_path.name}")
        print(f"{'='*70}")

        try:
            # Process with cleanup option
            test_ocr_pipeline(str(image_path), cleanup_input=cleanup_images)
            successful += 1
        except Exception as e:
            print(f"\n[ERROR] Failed to process page {idx}: {e}")
            failed += 1
            # Still try to cleanup if requested
            if cleanup_images:
                try:
                    image_path.unlink()
                    print(f"[CLEANUP] Deleted: {image_path}")
                except:
                    pass

    # Cleanup empty directory if all images were deleted
    if cleanup_images and image_paths:
        images_dir = image_paths[0].parent
        if images_dir.exists():
            try:
                # Only remove if empty
                if not list(images_dir.iterdir()):
                    images_dir.rmdir()
                    print(f"\n[CLEANUP] Removed empty directory: {images_dir}")
            except:
                pass

    # Final summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"\nTotal pages: {len(image_paths)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if cleanup_images:
        print(f"\nInput images: Cleaned up")
    else:
        print(f"\nInput images preserved in: {image_paths[0].parent if image_paths else 'N/A'}")

    print(f"\nOutput directory: output/")
    print(f"  - {successful} annotated images")
    print(f"  - {successful} markdown files")

    print("\n" + "="*70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/batch_process_pdf.py <path_to_pdf> [--keep-images]")
        print("\nOptions:")
        print("  --keep-images    Keep page images after processing (default: cleanup)")
        print("\nExample:")
        print("  uv run python scripts/batch_process_pdf.py input/document.pdf")
        print('  uv run python scripts/batch_process_pdf.py "C:\\Users\\name\\Downloads\\doc.pdf"')
        sys.exit(1)

    pdf_path = sys.argv[1]
    cleanup = "--keep-images" not in sys.argv

    batch_process_pdf(pdf_path, cleanup_images=cleanup)
