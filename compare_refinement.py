"""
Compare OCR extraction with and without Claude Sonnet 4.5 refinement.

This script orchestrates extraction, saves outputs, and shows comparison.
Useful for verifying the quality and impact of Claude refinement.

Usage:
    python compare_refinement.py input/Table.jpg
    python compare_refinement.py input/Table.jpg --output-dir custom_output
    python compare_refinement.py input/Table.jpg --render-html
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

from src.ocr_pipeline.qwen_extractor import extract_document


def save_text_file(content: str, filepath: Path):
    """Save text content to file with UTF-8 encoding."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Saved: {filepath}")


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np

    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_json_file(data: dict, filepath: Path):
    """Save JSON data to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types to native Python types
    serializable_data = convert_to_serializable(data)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {filepath}")


def render_html_to_file(html_content: str, output_path: Path):
    """
    Save HTML with proper document structure for viewing in browser.

    Args:
        html_content: HTML content (table, divs, etc.)
        output_path: Path to save HTML file
    """
    # Wrap content in full HTML document
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Output - {output_path.stem}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f4f4f4;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f9f9f9;
        }}
        .metadata {{
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="metadata">
        <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        <strong>File:</strong> {output_path.name}
    </div>

    {html_content}
</body>
</html>
"""

    save_text_file(full_html, output_path)


def compare_outputs(baseline_result: dict, refined_result: dict) -> dict:
    """
    Compare baseline and refined extraction results.

    Args:
        baseline_result: Result from extract_document(refine=False)
        refined_result: Result from extract_document(refine=True)

    Returns:
        Dict with comparison metrics
    """
    comparison = {
        'baseline': {
            'markdown_length': len(baseline_result.get('markdown', '')),
            'elements_count': len(baseline_result.get('elements', [])),
            'images_count': len(baseline_result.get('images', [])),
            'success': baseline_result.get('success', False),
        },
        'refined': {
            'markdown_length': len(refined_result.get('markdown', '')),
            'elements_count': len(refined_result.get('elements', [])),
            'images_count': len(refined_result.get('images', [])),
            'success': refined_result.get('success', False),
            'refinement_applied': refined_result.get('refinement', {}).get('refinement_applied', False),
            'validation_passed': refined_result.get('refinement', {}).get('validation_passed', False),
        },
        'changes': {
            'length_diff': len(refined_result.get('markdown', '')) - len(baseline_result.get('markdown', '')),
            'length_ratio': len(refined_result.get('markdown', '')) / max(len(baseline_result.get('markdown', '')), 1),
        }
    }

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description='Compare OCR extraction with and without Claude refinement',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'image',
        type=str,
        help='Path to input image'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_comparison',
        help='Directory to save outputs (default: output_comparison)'
    )

    parser.add_argument(
        '--render-html',
        action='store_true',
        help='Render HTML outputs as viewable HTML files'
    )

    args = parser.parse_args()

    # Validate input
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = output_dir / f"{image_path.stem}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OCR Extraction Comparison: Baseline vs Claude Refinement")
    print("=" * 80)
    print(f"\nInput image: {image_path}")
    print(f"Output directory: {session_dir}")
    print()

    # ============================================================
    # Step 1: Baseline Extraction (No Refinement)
    # ============================================================
    print("[1/2] Running baseline extraction (QwenVL only)...")
    print("-" * 80)

    baseline_result = extract_document(
        str(image_path),
        include_images=True,
        include_usage=True,
        refine=False
    )

    if not baseline_result['success']:
        print(f"Error in baseline extraction: {baseline_result['error']}")
        return 1

    # Save baseline outputs
    print("\nSaving baseline outputs...")
    baseline_dir = session_dir / "baseline"

    # Save markdown (with LaTeX tables)
    save_text_file(
        baseline_result['markdown'],
        baseline_dir / "output_markdown.md"
    )

    # Save metadata
    metadata = {
        'success': baseline_result['success'],
        'elements': baseline_result['elements'],
        'usage': baseline_result.get('usage', {}),
        'quality': baseline_result.get('quality', {}),
        'markdown_length': len(baseline_result['markdown']),
        'images_count': len(baseline_result['images']),
    }
    save_json_file(metadata, baseline_dir / "metadata.json")

    print(f"\nBaseline stats:")
    print(f"  Elements detected: {len(baseline_result['elements'])}")
    print(f"  Images extracted: {len(baseline_result['images'])}")
    print(f"  Markdown length: {len(baseline_result['markdown'])} chars")

    # ============================================================
    # Step 2: Refined Extraction (Claude Sonnet 4.5)
    # ============================================================
    print("\n" + "=" * 80)
    print("[2/2] Running refined extraction (QwenVL -> Claude Sonnet 4.5)...")
    print("-" * 80)

    refined_result = extract_document(
        str(image_path),
        include_images=True,
        include_usage=True,
        refine=True
    )

    if not refined_result['success']:
        print(f"Error in refined extraction: {refined_result['error']}")
        return 1

    # Save refined outputs
    print("\nSaving refined outputs...")
    refined_dir = session_dir / "refined"

    # Save HTML output
    save_text_file(
        refined_result['markdown'],
        refined_dir / "output.html"
    )

    # Optionally render as viewable HTML
    if args.render_html:
        render_html_to_file(
            refined_result['markdown'],
            refined_dir / "output_rendered.html"
        )

    # Save metadata
    refined_metadata = {
        'success': refined_result['success'],
        'elements': refined_result['elements'],
        'usage': refined_result.get('usage', {}),
        'quality': refined_result.get('quality', {}),
        'refinement': refined_result.get('refinement', {}),
        'html_length': len(refined_result['markdown']),
        'images_count': len(refined_result['images']),
    }
    save_json_file(refined_metadata, refined_dir / "metadata.json")

    print(f"\nRefined stats:")
    print(f"  Elements detected: {len(refined_result['elements'])}")
    print(f"  Images extracted: {len(refined_result['images'])}")
    print(f"  HTML length: {len(refined_result['markdown'])} chars")

    refinement = refined_result.get('refinement', {})
    if refinement.get('success'):
        if refinement.get('refinement_applied'):
            print(f"  Refinement: APPLIED")
            print(f"  Validation: {'PASSED' if refinement.get('validation_passed') else 'FAILED'}")
        else:
            print(f"  Refinement: No changes needed (extraction already accurate)")
    else:
        print(f"  Refinement: FAILED - {refinement.get('error', 'Unknown error')}")

    # ============================================================
    # Step 3: Comparison
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    comparison = compare_outputs(baseline_result, refined_result)

    print(f"\nBaseline (QwenVL only):")
    print(f"  Format: Markdown with LaTeX tables")
    print(f"  Length: {comparison['baseline']['markdown_length']:,} chars")
    print(f"  Elements: {comparison['baseline']['elements_count']}")

    print(f"\nRefined (QwenVL + Claude):")
    print(f"  Format: Semantic HTML")
    print(f"  Length: {comparison['refined']['markdown_length']:,} chars")
    print(f"  Elements: {comparison['refined']['elements_count']}")
    print(f"  Refinement applied: {comparison['refined']['refinement_applied']}")

    print(f"\nChanges:")
    print(f"  Length difference: {comparison['changes']['length_diff']:,} chars")
    print(f"  Length ratio: {comparison['changes']['length_ratio']:.2%}")

    # Save comparison
    save_json_file(comparison, session_dir / "comparison.json")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)
    print(f"\nBaseline outputs:")
    print(f"  {baseline_dir / 'output_markdown.md'}")
    print(f"  {baseline_dir / 'metadata.json'}")

    print(f"\nRefined outputs:")
    print(f"  {refined_dir / 'output.html'}")
    if args.render_html:
        print(f"  {refined_dir / 'output_rendered.html'} (viewable in browser)")
    print(f"  {refined_dir / 'metadata.json'}")

    print(f"\nComparison:")
    print(f"  {session_dir / 'comparison.json'}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)

    if args.render_html:
        print(f"\nTo view rendered HTML, open in browser:")
        print(f"  {refined_dir.resolve() / 'output_rendered.html'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
