"""Command-line interface for OCR Pipeline"""

import sys
from pathlib import Path

from .qwen_extractor import extract_document


def process_image(image_path: str, include_images: bool = True) -> dict:
    """
    Process a single image through the QwenVL extraction pipeline.

    Args:
        image_path: Path to the image file
        include_images: Whether to extract and embed images (default: True)

    Returns:
        Result dictionary from extract_document() with:
        - success: bool
        - markdown: str (with inline base64 images if include_images=True)
        - images: list of dicts with {type, base64, bbox}
        - elements: list of dicts with {type, bbox}
        - error: str (if success=False)
    """
    # Validate input
    if not Path(image_path).exists():
        return {'success': False, 'error': 'File not found', 'markdown': '', 'images': [], 'elements': []}

    # Extract document
    result = extract_document(image_path, include_images=include_images)

    return result


def main():
    """Main entry point for the CLI"""
    if len(sys.argv) < 2:
        print("OCR Pipeline - QwenVL Document Extraction")
        print("\nUsage:")
        print("  python -m ocr_pipeline.cli <image_path>")
        print("\nExample:")
        print("  python -m ocr_pipeline.cli input/document.png")
        print("\nReturns:")
        print("  JSON result with markdown, images, and elements")
        return 1

    image_path = sys.argv[1]

    print("=" * 80)
    print("OCR Pipeline - QwenVL Document Extraction")
    print("=" * 80)
    print(f"\nProcessing: {image_path}")
    print("\nCalling QwenVL API...")

    result = process_image(image_path)

    if not result['success']:
        print(f"\nError: {result['error']}")
        return 1

    # Display statistics
    print(f"\nDetected elements: {len(result['elements'])}")
    for i, elem in enumerate(result['elements'], 1):
        print(f"  {i}. {elem['type']} - bbox: {elem['bbox']}")

    print(f"\nExtracted images: {len(result['images'])}")
    print(f"\nMarkdown length: {len(result['markdown'])} characters")

    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)
    print("\nNote: Results returned in memory only (not saved to disk)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
