"""Command-line interface for OCR Pipeline"""

import sys
import argparse
from pathlib import Path

from .ocr_pipeline import OCRPipeline


def process_image(
    image_path: str,
    include_images: bool = True,
    convert_tables_to_html: bool = False,
    refine: bool = False,
    agentic_refine: bool = False,
    max_iterations: int = 2,
    output_dir: str = None
) -> dict:
    """
    Process a single image through the QwenVL extraction pipeline.

    Args:
        image_path: Path to the image file
        include_images: Whether to extract and embed images (default: True)
        convert_tables_to_html: Whether to convert LaTeX tables to HTML
        refine: Whether to refine extraction using Claude Sonnet 4.5 (OLMoCR-2 approach)
        agentic_refine: Whether to use iterative agentic refinement (multiple iterations)
        max_iterations: Maximum refinement iterations (default: 2)

    Returns:
        Result dictionary from OCRPipeline.apply() with:
        - success: bool
        - markdown: str (with inline base64 images if include_images=True)
                    HTML format if refine=True, otherwise markdown with LaTeX
        - images: list of dicts with {type, base64, bbox}
        - elements: list of dicts with {type, bbox}
        - refinement: dict with refinement metadata (if refine=True)
        - error: str (if success=False)
    """
    # Validate input
    if not Path(image_path).exists():
        return {'success': False, 'error': 'File not found', 'markdown': '', 'images': [], 'elements': []}

    # Create pipeline and extract document
    pipeline = OCRPipeline(
        preprocess=True,  # Enable preprocessing by default
        refine=refine,
        agentic_refine=agentic_refine,
        max_refinement_iterations=max_iterations,
        convert_tables_to_html=convert_tables_to_html,
        output_dir=output_dir
    )

    result = pipeline.apply(
        image_path,
        include_images=include_images
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
  python -m ocr_pipeline.cli document.png --refine
  python -m ocr_pipeline.cli invoice.png --refine --html-tables

Features:
  --refine:      Uses Claude Sonnet 4.5 to visually verify and refine extraction
                 (OLMoCR-2 approach). Converts LaTeX→HTML, then Claude refines for
                 accuracy. Cost: +$0.04-0.12 per image. Recommended for production.

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

    parser.add_argument(
        '--refine',
        action='store_true',
        help='Refine extraction using Claude Sonnet 4.5 (OLMoCR-2 approach). Converts LaTeX→HTML, then Claude refines for accuracy. Cost: +$0.04-0.12 per image.'
    )

    parser.add_argument(
        '--agentic-refine',
        action='store_true',
        help='Use iterative agentic refinement (Claude reviews its own output multiple times). Cost: 2-3x single-pass refinement.'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=2,
        help='Maximum refinement iterations for agentic refinement (default: 2)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save QwenVL output and each iteration output (e.g., output/). If not specified, nothing is saved.'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OCR Pipeline - QwenVL Document Extraction")
    print("=" * 80)
    print(f"\nProcessing: {args.image}")
    if args.agentic_refine:
        print(f"Refinement: Agentic mode (up to {args.max_iterations} iterations)")
    elif args.refine:
        print("Refinement: Enabled (Claude Sonnet 4.5 visual verification)")
    if args.html_tables:
        print("Table conversion: Enabled (LaTeX → HTML via Gemini 2.5 Flash)")
    print("\nCalling QwenVL API...")

    result = process_image(
        args.image,
        convert_tables_to_html=args.html_tables,
        refine=args.refine,
        agentic_refine=args.agentic_refine,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir
    )

    if not result['success']:
        print(f"\nError: {result['error']}")
        return 1

    # Display statistics
    print(f"\nDetected elements: {len(result['elements'])}")
    for i, elem in enumerate(result['elements'], 1):
        print(f"  {i}. {elem['type']} - bbox: {elem['bbox']}")

    print(f"\nExtracted images: {len(result['images'])}")

    # Display refinement stats if enabled
    if 'refinement' in result:
        refinement = result['refinement']
        if refinement.get('success'):
            # Agentic refinement stats
            if refinement.get('agentic'):
                print(f"\n[SUCCESS] Agentic refinement complete")
                print(f"  Iterations: {refinement.get('iterations', 0)}")

                # Show stopping reason
                stopping_reason = refinement.get('stopping_reason', 'unknown')
                reason_display = {
                    'converged': 'Converged (no changes detected)',
                    'max_iterations_reached': 'Max iterations reached',
                    'api_error': 'API error',
                    'validation_failed': 'Validation failed'
                }
                print(f"  Stopped: {reason_display.get(stopping_reason, stopping_reason)}")

                if refinement.get('iteration_history'):
                    for iter_data in refinement['iteration_history']:
                        status = "[Changes made]" if iter_data.get('changes_made') else "[No changes]"
                        print(f"    Iteration {iter_data['iteration']}: {status}")
                if refinement.get('total_usage'):
                    usage = refinement['total_usage']
                    if 'total_tokens' in usage:
                        print(f"  Total tokens: {usage['total_tokens']}")
            # Single-pass refinement stats
            elif refinement.get('refinement_applied'):
                print("\n[SUCCESS] Claude refinement applied")
                if refinement.get('validation_passed'):
                    print("  Validation: PASSED")
            else:
                print("\n[SUCCESS] Claude verified extraction (no changes needed)")
        else:
            print(f"\n[WARNING] Refinement failed: {refinement.get('error', 'Unknown error')}")

    output_format = "HTML" if (args.refine or args.agentic_refine) else "Markdown with LaTeX tables"
    print(f"\nOutput format: {output_format}")
    print(f"Content length: {len(result['markdown'])} characters")

    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)
    print("\nNote: Results returned in memory only (not saved to disk)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
