"""
Test script for LayoutDetector - JSON output only

Tests layout detection on images from the input folder and saves results.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from src.ocr_pipeline.layout_detector import LayoutDetector


def create_annotated_image(image_path, elements, output_path):
    """
    Create an annotated image with bounding boxes drawn on it.

    Args:
        image_path: Path to original image
        elements: List of detected elements with bboxes
        output_path: Path to save annotated image
    """
    # Load image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Define colors for different element types
    color_map = {
        'header': '#FF0000',      # Red
        'paragraph': '#0000FF',   # Blue
        'table': '#00FF00',       # Green
        'handwritten': '#FF00FF', # Magenta
        'signature': '#FFA500',   # Orange
        'chart': '#00FFFF',       # Cyan
        'graph': '#00FFFF',       # Cyan
        'diagram': '#FFFF00',     # Yellow
        'infographic': '#FF1493', # Deep Pink
        'figure': '#8B4513',      # Brown
        'image': '#8B4513',       # Brown
        'check': '#FF4500',       # Orange Red
        'cheque': '#FF4500',      # Orange Red
        'list': '#9370DB',        # Purple
        'printed': '#4169E1',     # Royal Blue
    }

    default_color = '#808080'  # Gray for unknown types

    # Draw bounding boxes
    for idx, elem in enumerate(elements, 1):
        bbox = elem['bbox']
        elem_type = elem['type']
        color = color_map.get(elem_type, default_color)

        # Draw rectangle
        draw.rectangle(
            [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
            outline=color,
            width=3
        )

        # Draw label with element number and type
        label = f"{idx}. {elem_type}"

        # Try to use a default font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Get text bounding box for background
        try:
            text_bbox = draw.textbbox((bbox[0], bbox[1] - 25), label, font=font)
            # Draw background for text
            draw.rectangle(text_bbox, fill=color)
            # Draw text
            draw.text((bbox[0], bbox[1] - 25), label, fill='white', font=font)
        except:
            # Fallback if textbbox not available (older PIL versions)
            draw.text((bbox[0], bbox[1] - 25), label, fill=color, font=font)

    # Save annotated image
    image.save(output_path)
    return output_path


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
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Enable gap analysis and self-critique refinement"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Apply image preprocessing (CLAHE + sharpening) for better detection"
    )
    parser.add_argument(
        "--show-preprocessing",
        action="store_true",
        help="Save preprocessed image for inspection"
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
            # Show what features are enabled
            features = []
            if args.preprocess:
                features.append("preprocessing")
            if args.refine:
                features.append("gap refinement")

            feature_str = " + ".join(features) if features else "standard"
            print(f"[DETECTING] Analyzing layout with {feature_str}...")

            # Save preprocessed image if requested
            if args.preprocess and args.show_preprocessing:
                from src.ocr_pipeline.image_preprocessor import ImagePreprocessor
                preprocessor = ImagePreprocessor()
                preprocessed_img = preprocessor.preprocess_for_ocr(
                    image_path,
                    deskew=False,
                    enhance_contrast=False,
                    clahe=True,
                    sharpen=True,
                    denoise=False,
                    upscale=1.0
                )
                preprocessed_path = output_dir / f"{image_path.stem}_preprocessed.png"
                preprocessed_img.save(preprocessed_path)
                print(f"[SAVED] Preprocessed image saved to: {preprocessed_path}")

            start_time = datetime.now()

            if args.refine:
                result = detector.detect_layout_with_refinement(image_path, preprocess=args.preprocess)
            else:
                result = detector.detect_layout(image_path, preprocess=args.preprocess)

            elapsed = (datetime.now() - start_time).total_seconds()

            # Display summary
            elements = result['elements']
            print(f"[SUCCESS] Detection complete in {elapsed:.2f}s")

            # Show refinement stats if available
            if 'refinement_stats' in result:
                stats = result['refinement_stats']
                print(f"\n[REFINEMENT] Stats:")
                print(f"   - Initial detection: {stats['initial_count']} elements")
                print(f"   - Gaps found: {stats['gaps_found']}")
                print(f"   - Gap elements found: {stats['gap_elements_found']}")
                print(f"   - Final count: {stats['final_count']} elements")
                print(f"   - Elements added: {stats['elements_added']}")

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

            # Add refinement stats if available
            if 'refinement_stats' in result:
                output_data['refinement_stats'] = result['refinement_stats']

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"\n[SAVED] Results saved to: {output_file}")

            # Create annotated image with bounding boxes
            print("[VISUALIZING] Creating annotated image with bounding boxes...")
            annotated_path = output_dir / f"{image_path.stem}_annotated.png"
            create_annotated_image(image_path, elements, annotated_path)
            print(f"[SAVED] Annotated image saved to: {annotated_path}")

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
