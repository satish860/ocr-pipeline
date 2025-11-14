"""
Test script for LayoutDetector - JSON output only

Tests layout detection on images from the input folder and saves results.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from src.ocr_pipeline.layout_detector import LayoutDetector


def main():
    """Run layout detection tests"""
    parser = argparse.ArgumentParser(description="Test LayoutDetector with JSON output")
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number to test (1-9, default: 1)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all pages in the input folder"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input/2e1b63c5-761d-48b9-b3b5-f263c3db4e30_images",
        help="Input directory containing test images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize detector
    print("Initializing LayoutDetector...")
    detector = LayoutDetector()

    # Get list of images to process
    if args.all:
        image_files = sorted(input_dir.glob("page_*.png"))
        if not image_files:
            print(f"[ERROR] No images found in {input_dir}")
            return
        print(f"\n[INFO] Found {len(image_files)} images to process\n")
    else:
        page_num = args.page
        image_path = input_dir / f"page_{page_num:04d}.png"
        if not image_path.exists():
            print(f"[ERROR] Image not found: {image_path}")
            return
        image_files = [image_path]

    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        print(f"{'='*60}")
        print(f"Processing [{idx}/{len(image_files)}]: {image_path.name}")
        print(f"{'='*60}")

        try:
            # Detect layout
            print("[DETECTING] Analyzing layout elements...")
            start_time = datetime.now()

            result = detector.detect_layout(image_path)

            elapsed = (datetime.now() - start_time).total_seconds()

            # Display summary
            elements = result['elements']
            print(f"[SUCCESS] Detection complete in {elapsed:.2f}s")
            print(f"\n[SUMMARY] Detected {len(elements)} elements:")

            # Group by type
            type_counts = {}
            for elem in elements:
                elem_type = elem['type']
                type_counts[elem_type] = type_counts.get(elem_type, 0) + 1

            for elem_type, count in sorted(type_counts.items()):
                print(f"   - {elem_type}: {count}")

            # Display first few elements
            print(f"\n[ELEMENTS] Sample elements (first 5):")
            for elem in elements[:5]:
                bbox = elem['bbox']
                print(f"   - {elem['type']}: bbox=[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")

            if len(elements) > 5:
                print(f"   ... and {len(elements) - 5} more")

            # Save results
            output_file = output_dir / f"{image_path.stem}_layout.json"

            # Prepare output data
            output_data = {
                "filename": image_path.name,
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": elapsed,
                "image_dimensions": result['image_dimensions'],
                "total_elements": len(elements),
                "elements_by_type": type_counts,
                "elements": elements
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"\n[SAVED] Results saved to: {output_file}")

        except Exception as e:
            print(f"[ERROR] Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if idx < len(image_files):
            print()  # Empty line between images

    print(f"\n{'='*60}")
    print(f"[COMPLETE] Processing complete! Processed {len(image_files)} image(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
