"""OCR Pipeline Test - End-to-End Testing"""

import sys
from pathlib import Path
from PIL import ImageDraw, ImageFont

from src.ocr_pipeline.image_preprocessor import ImagePreprocessor
from src.ocr_pipeline.layout_detector import LayoutDetector
from src.ocr_pipeline.region_extractor import RegionExtractor
from src.ocr_pipeline.ocr_extractor import OCRExtractor
from src.ocr_pipeline.spatial_analyzer import SpatialAnalyzer


# Configuration
OUTPUT_DIR = Path("output")
BBOX_COLORS = {
    'table': '#FF0000',
    'paragraph': '#00FF00',
    'header': '#FF00FF',
    'handwritten': '#FF1493',
}


def test_ocr_pipeline(image_path: str, cleanup_input: bool = False):
    """
    Complete OCR pipeline test:
    1. Preprocess image (deskew)
    2. Detect layout with QwenVL
    3. Extract regions
    4. OCR each region to markdown
    4.5. Analyze spatial relationships
    5. Generate spatially-aware markdown
    6. Visualize results
    7. Optionally cleanup input image (useful for PDF-to-image conversions)

    Args:
        image_path: Path to input image
        cleanup_input: Delete input image after processing (default: False)
    """
    print("\n" + "="*60)
    print("OCR PIPELINE TEST")
    print("="*60)
    print(f"\nInput: {image_path}")

    # Validate input
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        print(f"\n[ERROR] Image not found: {image_path}")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: Preprocess - Deskew image
    print("\n[1/5] Preprocessing: Deskewing image...")
    preprocessor = ImagePreprocessor()
    deskewed_image, angle = preprocessor.deskew_image(image_path)
    print(f"      Rotation correction: {angle:.2f} degrees")

    # Step 2: Detect Layout with QwenVL
    print("\n[2/5] Layout Detection: Running QwenVL...")
    detector = LayoutDetector()
    layout_result = detector.detect_layout(deskewed_image, output_format="json")
    print(f"      Detected {len(layout_result['elements'])} elements")

    # Step 3: Extract Regions
    print("\n[3/5] Region Extraction: Extracting bounding boxes...")
    extractor = RegionExtractor()
    regions = extractor.extract_regions(deskewed_image, layout_result)
    print(f"      Extracted {len(regions)} regions")

    # Step 4: OCR Extraction - Convert regions to markdown
    print("\n[4/5] OCR Extraction: Converting regions to markdown...")
    ocr = OCRExtractor()
    markdown_outputs = []

    for region in regions:
        region_index = region['index']
        region_type = region['type']
        region_bbox = region['bbox']
        region_image = region['image']

        print(f"      Processing region {region_index}/{len(regions)} ({region_type})...")

        try:
            # Extract text with context (Option B: save chart images)
            markdown_text = ocr.extract_text(
                region_image,
                element_type=region_type,
                output_dir=str(OUTPUT_DIR),
                region_index=region_index
            )

            # Store for combined output
            markdown_outputs.append({
                'index': region_index,
                'type': region_type,
                'bbox': region_bbox,
                'markdown': markdown_text
            })

        except Exception as e:
            print(f"      [WARNING] Failed to extract region {region_index}: {e}")
            markdown_outputs.append({
                'index': region_index,
                'type': region_type,
                'bbox': region_bbox,
                'markdown': f"[Error extracting text: {e}]"
            })

    print(f"      Extracted text from {len(markdown_outputs)} regions")

    # Step 4.5: Spatial Analysis - Understand region relationships
    print("\n[4.5/5] Spatial Analysis: Analyzing region relationships...")
    spatial_analyzer = SpatialAnalyzer()

    # Merge markdown_outputs back into regions for analysis
    for i, region in enumerate(regions):
        matching_md = next((md for md in markdown_outputs if md['index'] == region['index']), None)
        if matching_md:
            region['markdown'] = matching_md['markdown']

    # Analyze relationships
    regions_with_relationships = spatial_analyzer.analyze_relationships(regions)

    # Group by vertical alignment (useful for tables/rows)
    aligned_groups = spatial_analyzer.group_by_vertical_alignment(regions_with_relationships)
    print(f"      Found {len(aligned_groups)} vertically-aligned groups")

    # Create combined markdown document with spatial awareness
    output_name = image_path_obj.stem
    combined_md_path = OUTPUT_DIR / f"{output_name}_complete.md"

    with open(combined_md_path, 'w', encoding='utf-8') as f:
        # Generate spatially-aware markdown
        processed_indices = set()

        for group in aligned_groups:
            # Check if this group has mixed content (e.g., table + handwritten)
            if len(group) > 1:
                # Check types in group
                types = [r['type'] for r in group]

                # If we have table + handwritten on same row, merge them
                if 'table' in types or 'paragraph' in types:
                    main_region = group[0]  # Leftmost region
                    related_regions = group[1:]  # Right-side regions

                    f.write(f"<!-- Region {main_region['index']}: {main_region['type']} {main_region['bbox']} -->\n\n")
                    f.write(main_region.get('markdown', ''))

                    # Add related content as annotations
                    if related_regions:
                        f.write("\n\n**Related content (spatial):**\n")
                        for rel in related_regions:
                            f.write(f"- *{rel['type']}*: {rel.get('markdown', '')}\n")

                    f.write("\n\n---\n\n")

                    # Mark all as processed
                    for r in group:
                        processed_indices.add(r['index'])
                else:
                    # Different types, output separately
                    for region in group:
                        if region['index'] not in processed_indices:
                            f.write(f"<!-- Region {region['index']}: {region['type']} {region['bbox']} -->\n\n")
                            f.write(region.get('markdown', ''))
                            f.write("\n\n---\n\n")
                            processed_indices.add(region['index'])
            else:
                # Single region in group
                region = group[0]
                if region['index'] not in processed_indices:
                    f.write(f"<!-- Region {region['index']}: {region['type']} {region['bbox']} -->\n\n")
                    f.write(region.get('markdown', ''))
                    f.write("\n\n---\n\n")
                    processed_indices.add(region['index'])

    print(f"      Saved spatially-aware markdown: {combined_md_path}")

    # Step 5: Visualize Results
    print("\n[5/5] Visualization: Creating annotated image...")

    # Draw bounding boxes
    draw = ImageDraw.Draw(deskewed_image)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()

    for idx, element in enumerate(layout_result['elements'], 1):
        bbox = element['bbox']
        elem_type = element['type']
        color = BBOX_COLORS.get(elem_type, '#FFFFFF')

        x1, y1, x2, y2 = bbox

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=8)

        # Draw label
        label_bg = [x1, y1 - 50, x1 + 400, y1]
        draw.rectangle(label_bg, fill=color)
        draw.text((x1 + 10, y1 - 45), f"{idx}: {elem_type}", fill='black', font=font)

    # Save outputs
    output_name = image_path_obj.stem
    annotated_path = OUTPUT_DIR / f"{output_name}_annotated.png"

    deskewed_image.save(annotated_path)

    print(f"      Saved annotated image: {annotated_path}")

    # Print Summary
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)

    # Count by type
    element_counts = {}
    for element in layout_result['elements']:
        elem_type = element['type']
        element_counts[elem_type] = element_counts.get(elem_type, 0) + 1

    print(f"\nTotal elements: {len(layout_result['elements'])}")
    print("\nBreakdown by type:")
    for elem_type, count in sorted(element_counts.items()):
        color = BBOX_COLORS.get(elem_type, 'N/A')
        print(f"  {elem_type:15} {count:3}  (Color: {color})")

    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("  - Annotated image with bounding boxes")
    print(f"  - Combined markdown: {combined_md_path.name}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

    # Optional cleanup: Delete input image after processing
    if cleanup_input:
        try:
            image_path_obj.unlink()
            print(f"\n[CLEANUP] Deleted input image: {image_path}")
        except Exception as e:
            print(f"\n[WARNING] Failed to delete input image: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python test_layout_detector.py <image_path>")
        print("\nExample:")
        print("  uv run python test_layout_detector.py input/2e1b63c5-761d-48b9-b3b5-f263c3db4e30_images/page_0004.png")
        sys.exit(1)

    image_path = sys.argv[1]
    test_ocr_pipeline(image_path)
