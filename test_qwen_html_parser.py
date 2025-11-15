"""
Standalone test script for QwenVL Markdown Document Parsing via OpenRouter.

This script tests the "qwenvl markdown" prompt approach from the Qwen3-VL cookbook
using OpenRouter's API with the qwen/qwen3-vl-235b-a22b-instruct model.
"""

import os
import base64
import re
import requests
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def encode_image_base64(image_path: str) -> str:
    """
    Encode image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def smart_resize(height: int, width: int, min_pixels: int = 512*32*32,
                 max_pixels: int = 2048*32*32, factor: int = 32) -> tuple[int, int]:
    """
    Calculate optimal resize dimensions maintaining aspect ratio.

    Args:
        height: Original image height
        width: Original image width
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels
        factor: Dimension must be multiple of this factor

    Returns:
        Tuple of (new_height, new_width)
    """
    total_pixels = height * width

    if total_pixels < min_pixels:
        scale = (min_pixels / total_pixels) ** 0.5
    elif total_pixels > max_pixels:
        scale = (max_pixels / total_pixels) ** 0.5
    else:
        scale = 1.0

    new_height = int(height * scale)
    new_width = int(width * scale)

    # Round to nearest factor
    new_height = (new_height // factor) * factor
    new_width = (new_width // factor) * factor

    return new_height, new_width


def call_qwen_markdown_api(image_path: str, prompt: str = "qwenvl markdown",
                           min_pixels: int = 512*32*32, max_pixels: int = 2048*32*32) -> str:
    """
    Call OpenRouter API with Qwen3-VL model to get Markdown output.

    Args:
        image_path: Path to the image file
        prompt: Prompt to send (default: "qwenvl markdown")
        min_pixels: Minimum pixels for image resize
        max_pixels: Maximum pixels for image resize

    Returns:
        Markdown response string with coordinate annotations
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    # Load and encode image
    base64_image = encode_image_base64(image_path)

    # Get image dimensions for smart resize info
    with Image.open(image_path) as img:
        width, height = img.size
        input_height, input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
        print(f"Original size: {width}x{height}")
        print(f"Smart resize: {input_width}x{input_height}")
        print(f"Pixel range: {min_pixels} - {max_pixels}")

    # Determine image format
    image_format = Path(image_path).suffix.lower().replace(".", "")
    if image_format == "jpg":
        image_format = "jpeg"

    # Prepare API request
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen/qwen3-vl-235b-a22b-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

    print(f"Calling OpenRouter API with model: qwen/qwen3-vl-235b-a22b-instruct")
    print(f"Prompt: '{prompt}'")

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def convert_bbox_to_pixels(bbox: tuple[int, int, int, int],
                          image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """
    Convert relative bbox coordinates (0-1000) to pixel coordinates.

    Args:
        bbox: Tuple of (x1, y1, x2, y2) in 0-1000 scale
        image_width: Original image width in pixels
        image_height: Original image height in pixels

    Returns:
        Tuple of (x1, y1, x2, y2) in pixel coordinates
    """
    x1, y1, x2, y2 = bbox

    px1 = int(x1 / 1000 * image_width)
    py1 = int(y1 / 1000 * image_height)
    px2 = int(x2 / 1000 * image_width)
    py2 = int(y2 / 1000 * image_height)

    # Ensure correct order
    if px1 > px2:
        px1, px2 = px2, px1
    if py1 > py2:
        py1, py2 = py2, py1

    return px1, py1, px2, py2


def clean_markdown_wrapper(content: str) -> str:
    """
    Remove code fence wrappers from markdown content.

    QwenVL often returns: ```markdown\n...\n```
    We need to strip these for proper markdown rendering.

    Args:
        content: Raw content from API

    Returns:
        Clean markdown content
    """
    # Remove opening ```markdown code fence
    content = re.sub(r'^```(?:markdown|html)\s*\n', '', content.strip())

    # Remove closing ```
    content = re.sub(r'\n```\s*$', '', content)

    return content.strip()


def parse_markdown_bboxes(markdown_content: str) -> list[dict]:
    """
    Parse markdown and extract coordinate comments.

    Format: <!-- Image (x1, y1, x2, y2) --> or <!-- Table (x1, y1, x2, y2) --> or <!-- Paragraph (x1, y1, x2, y2) -->

    Args:
        markdown_content: Markdown string with coordinate comments

    Returns:
        List of dicts with element info: {type, bbox}
    """
    # Only parse the element types we care about
    pattern = r"<!-- (Image|Table|Chart|Signature|Handwritten|Paragraph) \(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\) -->"
    matches = re.findall(pattern, markdown_content)

    elements = []
    for match in matches:
        element_type, x1, y1, x2, y2 = match
        elements.append({
            'type': element_type,
            'bbox': (int(x1), int(y1), int(x2), int(y2))
        })

    return elements


def detect_alignment(x1: int, x2: int, width: int = 1000) -> str:
    """
    Detect text alignment based on bounding box x-coordinates.

    Args:
        x1: Left x-coordinate (0-1000 scale)
        x2: Right x-coordinate (0-1000 scale)
        width: Total width (default 1000 for 0-1000 scale)

    Returns:
        'left', 'center', or 'right'
    """
    center_x = (x1 + x2) / 2

    # Right-aligned: center is in right third
    if center_x > width * 0.66:
        return 'right'
    # Center-aligned: center is in middle third
    elif center_x > width * 0.33:
        return 'center'
    # Left-aligned: center is in left third
    else:
        return 'left'


def add_alignment_to_markdown(markdown_content: str) -> str:
    """
    Post-process markdown to add HTML alignment tags based on coordinate annotations.

    Args:
        markdown_content: Markdown with coordinate annotations

    Returns:
        Markdown with HTML alignment tags added
    """
    lines = markdown_content.split('\n')
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this line is a coordinate annotation
        match = re.match(r'<!-- (Image|Table|Chart|Signature|Handwritten) \((\d+), (\d+), (\d+), (\d+)\) -->', line)

        if match:
            elem_type, x1, x2, y1, y2 = match.groups()
            x1, x2 = int(x1), int(x2)

            # Detect alignment
            alignment = detect_alignment(x1, x2)

            # Add the coordinate comment
            result_lines.append(line)
            i += 1

            # Collect lines until next coordinate annotation or empty line
            content_lines = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith('<!--'):
                content_lines.append(lines[i])
                i += 1

            # Wrap content with alignment if not left-aligned
            if content_lines:
                if alignment == 'right':
                    result_lines.append('<div align="right">')
                    result_lines.extend(content_lines)
                    result_lines.append('</div>')
                elif alignment == 'center':
                    result_lines.append('<div align="center">')
                    result_lines.extend(content_lines)
                    result_lines.append('</div>')
                else:
                    # Left-aligned, no tags needed
                    result_lines.extend(content_lines)
        else:
            result_lines.append(line)
            i += 1

    return '\n'.join(result_lines)


def visualize_markdown_bboxes(image_path: str, markdown_content: str, output_path: str = None):
    """
    Visualize bounding boxes from markdown coordinate comments.

    Args:
        image_path: Path to the original image
        markdown_content: Markdown string with coordinate comments
        output_path: Path to save annotated image (optional)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Parse markdown elements
    elements = parse_markdown_bboxes(markdown_content)

    print(f"\nFound {len(elements)} elements with coordinates:")
    for i, elem in enumerate(elements, 1):
        print(f"  {i}. {elem['type']} - bbox: {elem['bbox']}")

    if not elements:
        print("  No coordinate annotations found in markdown!")

    # Setup drawing
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    # Draw bounding boxes
    colors = {
        'Image': 'blue',
        'Table': 'red',
        'Chart': 'cyan',
        'Signature': 'magenta',
        'Handwritten': 'yellow',
        'Paragraph': 'green',
        'Heading': 'purple',
        'Text': 'orange'
    }

    for elem in elements:
        # Convert coordinates
        x1, y1, x2, y2 = convert_bbox_to_pixels(elem['bbox'], width, height)

        # Determine color
        color = colors.get(elem['type'], 'orange')

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        # Draw label
        label = elem['type']
        draw.text((x1, y1 - 20), label, fill=color, font=font)

    # Save or show
    if output_path:
        image.save(output_path)
        print(f"\nAnnotated image saved to: {output_path}")
    else:
        image.show()


def visualize_all_elements(image_path: str, elements: list, output_path: str = None,
                          resized_width: int = None, resized_height: int = None):
    """
    Visualize all detected elements (paragraphs, headings, etc.) with bounding boxes.

    Args:
        image_path: Path to the original image
        elements: List of tuples (element_type, x1, y1, x2, y2) from regex matches
        output_path: Path to save annotated image (optional)
        resized_width: Width of the image used by the model (if different from original)
        resized_height: Height of the image used by the model (if different from original)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size

    # If model used a resized image, we need to scale coordinates accordingly
    # Coordinates are 0-1000 relative to the RESIZED image dimensions
    if resized_width and resized_height:
        width, height = resized_width, resized_height
        print(f"  Using resized dimensions for coordinate conversion: {width}x{height}")
        print(f"  Original image dimensions: {orig_width}x{orig_height}")
    else:
        width, height = orig_width, orig_height

    # Setup drawing
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Color mapping for different element types
    color_map = {
        'Paragraph': 'green',
        'Heading': 'blue',
        'Title': 'purple',
        'Text': 'orange',
        'Table': 'red',
        'Image': 'cyan',
        'List': 'yellow',
        'Header': 'magenta',
        'Footer': 'brown',
        'default': 'gray'
    }

    print(f"  Drawing {len(elements)} bounding boxes...")

    for elem_type, x1, y1, x2, y2 in elements:
        # Convert relative coordinates (0-1000) to pixels on resized image
        px1, py1, px2, py2 = convert_bbox_to_pixels(
            (int(x1), int(y1), int(x2), int(y2)), width, height
        )

        # If we're working with resized dimensions, scale back to original
        if resized_width and resized_height:
            scale_x = orig_width / resized_width
            scale_y = orig_height / resized_height
            px1 = int(px1 * scale_x)
            py1 = int(py1 * scale_y)
            px2 = int(px2 * scale_x)
            py2 = int(py2 * scale_y)

        # Get color for this element type
        color = color_map.get(elem_type, color_map['default'])

        # Draw rectangle
        draw.rectangle([px1, py1, px2, py2], outline=color, width=3)

        # Draw label with background
        label = elem_type
        label_bbox = draw.textbbox((px1, py1 - 20), label, font=small_font)
        label_bg = [label_bbox[0] - 2, label_bbox[1] - 2, label_bbox[2] + 2, label_bbox[3] + 2]
        draw.rectangle(label_bg, fill=color)
        draw.text((px1, py1 - 20), label, fill='white', font=small_font)

    # Save or show
    if output_path:
        image.save(output_path)
    else:
        image.show()


def main(image_path: str = None, clear_output: bool = True):
    """
    Main test function - tests Markdown format with table conversion.

    Args:
        image_path: Optional path to specific image. If None, uses first image in input/
        clear_output: Whether to clear the output folder before processing (default: True)
    """
    print("=" * 80)
    print("QwenVL Document Parsing Test - via OpenRouter")
    print("Testing 'qwenvl markdown' format")
    print("=" * 80)

    # Clear output folder (only if requested)
    output_dir = Path("output")
    if clear_output and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        print("\nCleared output folder")
    output_dir.mkdir(exist_ok=True)

    # Configure test image
    if image_path:
        test_image = image_path
        if not Path(test_image).exists():
            print(f"\nError: Image file not found: {test_image}")
            return
        print(f"\nUsing specified image: {test_image}")
    else:
        input_dir = Path("input")
        if input_dir.exists():
            image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
            if image_files:
                test_image = str(image_files[0])
                print(f"\nUsing image from input folder: {test_image}")
            else:
                print("\nNo images found in input/ folder.")
                print("Please add a test image to the input/ folder and run again.")
                return
        else:
            print("\nInput folder not found. Please create it and add a test image.")
            return

    # Get input filename without extension for output naming
    input_name = Path(test_image).stem

    try:
        # Test Markdown Format
        print("\n" + "=" * 80)
        print("QWENVL MARKDOWN FORMAT")
        print("=" * 80)

        # Enhanced prompt to explicitly request LaTeX tables with coordinates
        # Only request coordinates for non-text elements (images, tables, charts, signatures, handwritten)
        enhanced_prompt = """qwenvl markdown

Convert this document to markdown format with the following requirements:
- Extract ALL text content as markdown in natural reading order (top to bottom, left to right)
- Preserve text alignment using HTML tags:
  - Right-aligned text: <div align="right">text</div>
  - Center-aligned text: <div align="center">text</div>
  - Left-aligned text: no tags needed (default)
- Use <br> tags for line breaks within aligned sections
- Represent all tables in LaTeX format using \\begin{tabular} and \\end{tabular}
- Add coordinate annotations ONLY for these special elements using HTML comments (0-1000 scale):
  - Tables: <!-- Table (x1, y1, x2, y2) --> followed by the table content
  - Images: <!-- Image (x1, y1, x2, y2) --> followed by image description
  - Charts/Graphs: <!-- Chart (x1, y1, x2, y2) --> followed by chart description
  - Signatures: <!-- Signature (x1, y1, x2, y2) --> followed by the signature text/name
  - Handwritten text/notes/stamps: <!-- Handwritten (x1, y1, x2, y2) --> followed by the handwritten text
- CRITICAL: Place each coordinate annotation immediately BEFORE its content at the correct position in reading order
- The signature annotation must appear where the signature actually appears in the document, not at the top
- Do NOT add coordinate annotations for regular typed text, paragraphs, or headings
- IMPORTANT: Identify and annotate ALL handwritten elements including stamps, signatures, and handwritten notes
- For tables with shared/tied ranks or merged cells, repeat the value in all rows that share it
- Maintain spatial/positional order of all elements in the output
- Detect the horizontal position of text and apply appropriate alignment (right/center/left)
"""

        print("\nCalling API with 'qwenvl markdown' prompt (enhanced for OpenRouter)...")
        markdown_output_raw = call_qwen_markdown_api(test_image, prompt=enhanced_prompt)

        print("\n" + "-" * 80)
        print("RAW Markdown Response (before cleaning):")
        print("-" * 80)
        # Save raw output for inspection
        output_raw = Path("output") / f"{input_name}_qwen_raw_response.txt"
        output_raw.write_text(markdown_output_raw, encoding='utf-8')
        print(f"  Saved raw response to: {output_raw}")
        # Handle Unicode characters in preview
        try:
            print(markdown_output_raw[:1500])
        except UnicodeEncodeError:
            # If console can't handle Unicode, show length only
            print(f"[Content contains special characters, showing length only]")
        if len(markdown_output_raw) > 1500:
            print(f"\n... (truncated, total length: {len(markdown_output_raw)} chars)")

        # Clean markdown (remove code fences)
        print("\n" + "-" * 80)
        print("Cleaning Markdown Output:")
        print("-" * 80)
        markdown_output = clean_markdown_wrapper(markdown_output_raw)
        print(f"  Removed code fence wrappers")
        print(f"  Clean length: {len(markdown_output)} chars")

        # Add alignment tags based on coordinates (only if model didn't add them)
        print("\n" + "-" * 80)
        print("Adding Alignment Tags:")
        print("-" * 80)
        if '<div align=' in markdown_output:
            print(f"  Model already added alignment tags - skipping post-processing")
        else:
            markdown_output = add_alignment_to_markdown(markdown_output)
            print(f"  Analyzed coordinates and added HTML alignment tags")
        print(f"  Final length: {len(markdown_output)} chars")

        # Save original Markdown
        output_md = Path("output") / f"{input_name}_qwen_markdown_output.md"
        output_md.write_text(markdown_output, encoding='utf-8')
        print(f"  Saved markdown with alignment to: {output_md}")

        # Parse Markdown coordinate annotations (no visualization)
        print("\n" + "-" * 80)
        print("Parsing Markdown Coordinate Annotations:")
        print("-" * 80)
        elements = parse_markdown_bboxes(markdown_output)
        print(f"\nFound {len(elements)} elements with coordinates:")
        for i, elem in enumerate(elements, 1):
            print(f"  {i}. {elem['type']} - bbox: {elem['bbox']}")
        if not elements:
            print("  No coordinate annotations found in markdown!")

        # Extract and save cropped images/tables/signatures/charts only
        # Filter for specific element types we want to extract
        extract_types = {'Image', 'Table', 'Chart', 'Signature', 'Handwritten'}
        elements_to_extract = [elem for elem in elements if elem['type'] in extract_types]

        if elements_to_extract:
            print("\n" + "-" * 80)
            print("Extracting Images/Tables/Charts/Signatures/Handwritten:")
            print("-" * 80)

            # Load original image
            with Image.open(test_image) as img:
                width, height = img.size

                extract_count = 0
                for i, elem in enumerate(elements, 1):
                    elem_type = elem['type']

                    # Skip if not in extract types
                    if elem_type not in extract_types:
                        continue

                    extract_count += 1
                    bbox = elem['bbox']

                    # Convert 0-1000 coordinates to pixels
                    px1, py1, px2, py2 = convert_bbox_to_pixels(bbox, width, height)

                    # Crop the region
                    cropped = img.crop((px1, py1, px2, py2))

                    # Save cropped image
                    output_crop = Path("output") / f"{input_name}_{elem_type.lower()}_{extract_count}.png"
                    cropped.save(output_crop)

                    crop_width = px2 - px1
                    crop_height = py2 - py1
                    print(f"  {extract_count}. Extracted {elem_type} ({crop_width}x{crop_height}) -> {output_crop.name}")

            if extract_count == 0:
                print("  No extractable elements (Image/Table/Chart/Signature/Handwritten) found!")
        else:
            print("\n  No extractable elements (Image/Table/Chart/Signature/Handwritten) found!")

        # Summary
        print("\n" + "=" * 80)
        print("TEST COMPLETE!")
        print("=" * 80)
        print("\nFormat: 'qwenvl markdown' with table/image coordinates")
        print("\nOutput Files:")
        print(f"  - {output_md} (markdown with coordinates)")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    # Check for command-line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        main(image_path)
    else:
        main()
