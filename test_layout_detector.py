"""Test script for QwenVL layout detection"""

from src.ocr_pipeline.layout_detector import LayoutDetector
from src.ocr_pipeline.region_extractor import RegionExtractor
from src.ocr_pipeline.image_preprocessor import ImagePreprocessor
from PIL import Image, ImageDraw, ImageFont
import sys


def create_sample_document():
    """Create a simple test document image"""
    # Create a white image
    width, height = 800, 1000
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # Try to use a decent font, fallback to default
    try:
        title_font = ImageFont.truetype("arial.ttf", 40)
        header_font = ImageFont.truetype("arial.ttf", 28)
        text_font = ImageFont.truetype("arial.ttf", 18)
    except:
        # Fallback to default font
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    # Draw title
    draw.text((50, 50), "Sample Document", fill='black', font=title_font)

    # Draw section header
    draw.text((50, 150), "Introduction", fill='black', font=header_font)

    # Draw paragraph
    para_text = """This is a sample document for testing layout detection.
The layout detector should identify this as a paragraph.
It contains multiple lines of text."""
    draw.text((50, 200), para_text, fill='black', font=text_font)

    # Draw another section
    draw.text((50, 320), "Methods", fill='black', font=header_font)

    # Draw a "table" (just boxes with text)
    draw.rectangle([50, 370, 750, 550], outline='black', width=2)
    draw.line([50, 410, 750, 410], fill='black', width=2)
    draw.line([400, 370, 400, 550], fill='black', width=2)

    draw.text((60, 380), "Header 1", fill='black', font=text_font)
    draw.text((410, 380), "Header 2", fill='black', font=text_font)
    draw.text((60, 430), "Data 1", fill='black', font=text_font)
    draw.text((410, 430), "Data 2", fill='black', font=text_font)

    # Draw conclusion section
    draw.text((50, 600), "Conclusion", fill='black', font=header_font)
    para2_text = """This test document helps verify that the layout
detector can identify different document elements."""
    draw.text((50, 650), para2_text, fill='black', font=text_font)

    # Save the image
    image.save("test_document.png")
    print("[INFO] Created test_document.png")

    return image


def test_layout_detection():
    """Test layout detection with both HTML and JSON formats"""

    print("\n" + "="*60)
    print("TESTING QWENVL LAYOUT DETECTOR")
    print("="*60)

    # Create sample document
    print("\n[1] Creating sample document...")
    image = create_sample_document()

    # Initialize detector
    print("\n[2] Initializing LayoutDetector...")
    try:
        detector = LayoutDetector()
        print("[SUCCESS] LayoutDetector initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return

    # Test HTML format
    print("\n" + "-"*60)
    print("TESTING HTML FORMAT")
    print("-"*60)
    try:
        result = detector.detect_layout("test_document.png", output_format="html")

        print(f"\n[INFO] Format: {result['format']}")
        print(f"[INFO] Image dimensions: {result['image_dimensions']}")
        print(f"[INFO] Detected {len(result['elements'])} elements")

        print("\nDetected Elements (HTML):")
        for i, elem in enumerate(result['elements'], 1):
            print(f"\n  Element {i}:")
            print(f"    Type: {elem['type']}")
            if elem.get('class'):
                print(f"    Class: {elem['class']}")
            print(f"    BBox: {elem['bbox']}")
            print(f"    Content: {elem['content'][:80]}..." if len(elem['content']) > 80 else f"    Content: {elem['content']}")

        print("\n[HTML RAW OUTPUT PREVIEW]:")
        print(result['raw_output'][:500] + "..." if len(result['raw_output']) > 500 else result['raw_output'])

    except Exception as e:
        print(f"[ERROR] HTML format test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test JSON format
    print("\n" + "-"*60)
    print("TESTING JSON FORMAT")
    print("-"*60)
    try:
        result = detector.detect_layout("test_document.png", output_format="json")

        print(f"\n[INFO] Format: {result['format']}")
        print(f"[INFO] Image dimensions: {result['image_dimensions']}")
        print(f"[INFO] Detected {len(result['elements'])} elements")

        print("\nDetected Elements (JSON):")
        for i, elem in enumerate(result['elements'], 1):
            print(f"\n  Element {i}:")
            print(f"    Type: {elem['type']}")
            print(f"    BBox: {elem['bbox']}")
            print(f"    Content: {elem['content'][:80]}..." if len(elem['content']) > 80 else f"    Content: {elem['content']}")

        print("\n[JSON RAW OUTPUT PREVIEW]:")
        print(result['raw_output'][:500] + "..." if len(result['raw_output']) > 500 else result['raw_output'])

    except Exception as e:
        print(f"[ERROR] JSON format test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


def test_real_image():
    """Test with real image from input folder"""

    print("\n" + "="*60)
    print("TESTING WITH REAL IMAGE: input/image-1.jpg")
    print("="*60)

    # Initialize detector
    print("\n[1] Initializing LayoutDetector...")
    try:
        detector = LayoutDetector()
        print("[SUCCESS] LayoutDetector initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return

    # Test with JSON format
    print("\n[2] Running layout detection (JSON format)...")
    try:
        result = detector.detect_layout("input/image-1.jpg", output_format="json")

        print(f"\n[INFO] Format: {result['format']}")
        print(f"[INFO] Image dimensions: {result['image_dimensions']['width']} x {result['image_dimensions']['height']}")
        print(f"[INFO] Detected {len(result['elements'])} elements")

        print("\nDetected Elements:")
        for i, elem in enumerate(result['elements'], 1):
            print(f"\n  Element {i}:")
            print(f"    Type: {elem['type']}")
            print(f"    BBox: {elem['bbox']}")
            content_preview = elem['content'][:80] + "..." if len(elem['content']) > 80 else elem['content']
            print(f"    Content: {content_preview}")

        print("\n[RAW OUTPUT]:")
        print(result['raw_output'])

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("REAL IMAGE TEST COMPLETE")
    print("="*60)


def test_region_extraction():
    """Test region extraction pipeline"""

    print("\n" + "="*60)
    print("TESTING REGION EXTRACTION PIPELINE")
    print("="*60)

    image_path = "input/2e1b63c5-761d-48b9-b3b5-f263c3db4e30_images/page_0004.png"

    # Step 1: Layout Detection
    print("\n[1] Running layout detection...")
    detector = LayoutDetector()
    layout_result = detector.detect_layout(image_path, output_format="json")

    print(f"[INFO] Detected {len(layout_result['elements'])} elements")

    # Step 2: Region Extraction
    print("\n[2] Extracting regions...")
    extractor = RegionExtractor()
    regions = extractor.extract_regions(image_path, layout_result)

    print(f"[INFO] Extracted {len(regions)} regions")

    # Display region info
    print("\n" + "-"*60)
    print("EXTRACTED REGIONS:")
    print("-"*60)
    for region in regions:
        print(f"\nRegion {region['index']}:")
        print(f"  Type: {region['type']}")
        print(f"  BBox: {region['bbox']}")
        print(f"  Size: {region['width']}x{region['height']} pixels")
        try:
            content_preview = region['content_preview'][:60] + "..." if len(region['content_preview']) > 60 else region['content_preview']
            print(f"  Content Preview: {content_preview}")
        except UnicodeEncodeError:
            print(f"  Content Preview: [Contains non-ASCII characters]")

    # Step 3: Save regions
    print("\n[3] Saving regions to output directory...")
    saved_files = extractor.save_regions(regions, output_dir="output")

    print(f"\n[SUCCESS] Saved {len(saved_files)} region images")

    print("\n" + "="*60)
    print("REGION EXTRACTION TEST COMPLETE")
    print("="*60)
    print("\nCheck the 'output' folder for extracted region images!")


def test_deskew():
    """Test deskewing and layout detection on page 4"""

    print("\n" + "="*60)
    print("TESTING DESKEW ON PAGE 4")
    print("="*60)

    input_path = "input/2e1b63c5-761d-48b9-b3b5-f263c3db4e30_images/page_0004.png"

    # Step 1: Load and deskew image
    print("\n[1] Deskewing image...")
    preprocessor = ImagePreprocessor()
    deskewed_image, angle = preprocessor.deskew_image(input_path)

    # Save deskewed image
    deskewed_path = "output/page_0004_deskewed.png"
    deskewed_image.save(deskewed_path)
    print(f"[OK] Saved deskewed image to: {deskewed_path}")
    print(f"[INFO] Rotation correction: {angle:.2f} degrees")

    # Step 2: Run layout detection on deskewed image
    print("\n[2] Running layout detection on deskewed image...")
    detector = LayoutDetector()
    result = detector.detect_layout(deskewed_image, output_format="json")

    print(f"[OK] Detected {len(result['elements'])} elements")

    # Step 3: Visualize results
    print("\n[3] Creating visualization...")
    draw = ImageDraw.Draw(deskewed_image)

    # Define colors
    colors = {
        'table': '#FF0000',
        'paragraph': '#00FF00',
        'header': '#FF00FF',
        'handwritten': '#FF1493',
    }

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()

    # Draw bounding boxes
    for idx, element in enumerate(result['elements'], 1):
        bbox = element['bbox']
        elem_type = element['type']
        color = colors.get(elem_type, '#FFFFFF')

        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=8)

        # Draw label
        label_bg_coords = [x1, y1 - 50, x1 + 400, y1]
        draw.rectangle(label_bg_coords, fill=color)
        draw.text((x1 + 10, y1 - 45), f"{idx}: {elem_type}", fill='black', font=font)

    # Save annotated image
    annotated_path = "output/page_0004_deskewed_annotated.png"
    deskewed_image.save(annotated_path)
    print(f"[OK] Saved annotated image to: {annotated_path}")

    # Step 4: Print summary
    print("\n" + "="*60)
    print("DETECTION SUMMARY (After Deskew)")
    print("="*60)

    element_counts = {}
    for element in result['elements']:
        elem_type = element['type']
        element_counts[elem_type] = element_counts.get(elem_type, 0) + 1

    print(f"\nTotal elements detected: {len(result['elements'])}")
    print("\nBreakdown by type:")
    for elem_type, count in sorted(element_counts.items()):
        print(f"  {elem_type}: {count}")

    print("\n" + "="*60)
    print("FILES CREATED")
    print("="*60)
    print(f"  1. {deskewed_path}")
    print(f"  2. {annotated_path}")
    print("\nCompare with original:")
    print("  uv run python test_layout_detector.py --visualize")

    print("\n" + "="*60)
    print("DESKEW TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        test_real_image()
    elif len(sys.argv) > 1 and sys.argv[1] == "--extract":
        test_region_extraction()
    elif len(sys.argv) > 1 and sys.argv[1] == "--deskew":
        test_deskew()
    else:
        test_layout_detection()
        print("\n\nTips:")
        print("  --real    : Test with input/image-1.jpg")
        print("  --extract : Test region extraction pipeline")
        print("  --deskew  : Test deskewing on page 4 with visualization")
