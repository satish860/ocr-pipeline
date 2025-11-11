"""
Visualize layout detection results by drawing bounding boxes on the image.
Usage: uv run python scripts/visualize_layout.py <image_path> [output_path]
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path to import ocr_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline.layout_detector import LayoutDetector


def visualize_layout(image_path: str, output_path: str = None):
    """
    Detect layout and draw bounding boxes on the image.

    Args:
        image_path: Path to input image
        output_path: Path to save annotated image (default: <image_path>_annotated.png)
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Set default output path
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_annotated.png"
    else:
        output_path = Path(output_path)

    print(f"Processing: {image_path}")
    print(f"Output will be saved to: {output_path}")

    # Step 1: Detect layout
    print("\n[1] Running layout detection...")
    detector = LayoutDetector()
    result = detector.detect_layout(str(image_path), output_format="json")

    print(f"[OK] Detected {len(result['elements'])} elements")

    # Step 2: Load image and prepare drawing
    print("\n[2] Drawing bounding boxes...")
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Define colors for different element types
    colors = {
        'table': '#FF0000',        # Red
        'paragraph': '#00FF00',    # Green
        'header': '#FF00FF',       # Magenta
        'h1': '#0000FF',           # Blue
        'h2': '#FF00FF',           # Magenta
        'h3': '#00FFFF',           # Cyan
        'list': '#FFFF00',         # Yellow
        'figure': '#FFA500',       # Orange
        'handwritten': '#FF1493',  # Deep Pink (for handwritten)
    }

    # Try to load a font, fallback to default
    try:
        # Use larger font for better visibility
        font = ImageFont.truetype("arial.ttf", 40)
        small_font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Draw each element
    for idx, element in enumerate(result['elements'], 1):
        bbox = element['bbox']
        elem_type = element['type']

        # Get color for this element type
        color = colors.get(elem_type, '#FFFFFF')  # Default to white

        # Draw rectangle
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=8)

        # Draw label background (semi-transparent effect with rectangle)
        label_text = f"{idx}: {elem_type}"

        # Draw label background
        label_bg_coords = [x1, y1 - 50, x1 + 400, y1]
        draw.rectangle(label_bg_coords, fill=color)

        # Draw label text in black
        draw.text((x1 + 10, y1 - 45), label_text, fill='black', font=small_font)

    # Step 3: Save annotated image
    print("\n[3] Saving annotated image...")
    image.save(output_path)
    print(f"[OK] Saved to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)

    for idx, element in enumerate(result['elements'], 1):
        bbox = element['bbox']
        elem_type = element['type']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        print(f"\nRegion {idx}: {elem_type}")
        print(f"  BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        print(f"  Size: {width}x{height} pixels")

    print("\n" + "="*60)
    print(f"\nColor Legend:")
    for elem_type, color in colors.items():
        print(f"  {color} - {elem_type}")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/visualize_layout.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        visualize_layout(image_path, output_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
