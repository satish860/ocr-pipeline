"""Command-line interface for OCR Pipeline"""

import sys
import argparse
from pathlib import Path

from .qwen_extractor import extract_document


def process_image(
    image_path: str,
    include_images: bool = True,
    convert_tables_to_html: bool = False
) -> dict:
    """
    Process a single image through the QwenVL extraction pipeline.

    Args:
        image_path: Path to the image file
        include_images: Whether to extract and embed images (default: True)
        convert_tables_to_html: Whether to convert LaTeX tables to HTML using Gemini 2.5 Flash

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
    result = extract_document(
        image_path,
        include_images=include_images,
        convert_tables_to_html=convert_tables_to_html
    )

    return result


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(
        description='OCR Pipeline - QwenVL Document Extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ocr_pipeline.cli document.png
  python -m ocr_pipeline.cli document.png --html-tables
  python -m ocr_pipeline.cli invoice.png --html-tables

Features:
  --html-tables: Converts detected tables from LaTeX to HTML using Gemini 2.5 Flash
                 This improves accuracy by using both the table image and LaTeX
                 as input to generate clean, structured HTML tables.
        """
    )

    parser.add_argument(
        'image',
        type=str,
        help='Path to the image file to process'
    )

    parser.add_argument(
        '--html-tables',
        action='store_true',
        help='Convert tables to HTML using Gemini 2.5 Flash (improves accuracy)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OCR Pipeline - QwenVL Document Extraction")
    print("=" * 80)
    print(f"\nProcessing: {args.image}")
    if args.html_tables:
        print("Table conversion: Enabled (LaTeX → HTML via Gemini 2.5 Flash)")
    print("\nCalling QwenVL API...")

    result = process_image(
        args.image,
        convert_tables_to_html=args.html_tables
    )

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
