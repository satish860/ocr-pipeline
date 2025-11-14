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


def call_qwen_html_api(image_path: str, prompt: str = "qwenvl html",
                       min_pixels: int = 512*32*32, max_pixels: int = 2048*32*32) -> str:
    """
    Call OpenRouter API with Qwen3-VL model to get HTML output.

    Args:
        image_path: Path to the image file
        prompt: Prompt to send (default: "qwenvl html")
        min_pixels: Minimum pixels for image resize
        max_pixels: Maximum pixels for image resize

    Returns:
        HTML response string with data-bbox attributes
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


def latex_table_to_html(latex_table: str) -> str:
    """
    Convert LaTeX table format to HTML table.

    Args:
        latex_table: LaTeX table string (from \\begin{tabular} to \\end{tabular})

    Returns:
        HTML table string
    """
    # Extract table content between \begin{tabular} and \end{tabular}
    start = latex_table.find(r'\begin{tabular}')
    end = latex_table.find(r'\end{tabular}')

    if start == -1 or end == -1:
        return "<p>Invalid LaTeX table format</p>"

    # Get content after the column specification
    content_start = latex_table.find('}', start) + 1
    table_content = latex_table[content_start:end].strip()

    # Split by \hline to get rows
    rows = [row.strip() for row in table_content.split(r'\hline') if row.strip()]

    # Filter out column specification rows (contain only {|...|} pattern)
    rows = [row for row in rows if not (row.startswith('{') and row.endswith('}') and '&' not in row)]

    html_parts = ['<table border="1" style="border-collapse: collapse; width: 100%;">']

    is_first_row = True
    for row in rows:
        if not row or row == r'\\':
            continue

        # Remove trailing \\
        row = row.rstrip('\\').strip()

        # Split by & to get cells
        cells = [cell.strip() for cell in row.split('&')]

        # Determine if this is a header row (contains \textbf)
        is_header = is_first_row or any(r'\textbf' in cell for cell in cells)

        if is_header:
            html_parts.append('  <thead><tr>')
            tag = 'th'
        else:
            if is_first_row:
                html_parts.append('  <tbody>')
            html_parts.append('  <tr>')
            tag = 'td'

        for cell in cells:
            # Clean LaTeX formatting
            cleaned_cell = clean_latex_formatting(cell)
            html_parts.append(f'    <{tag}>{cleaned_cell}</{tag}>')

        if is_header:
            html_parts.append('  </tr></thead>')
            if is_first_row:
                html_parts.append('  <tbody>')
        else:
            html_parts.append('  </tr>')

        is_first_row = False

    html_parts.append('  </tbody>')
    html_parts.append('</table>')

    return '\n'.join(html_parts)


def clean_latex_formatting(text: str) -> str:
    """
    Clean LaTeX formatting from text.

    Args:
        text: Text with LaTeX formatting

    Returns:
        Plain text or HTML
    """
    # Remove \textbf{} and keep content
    text = re.sub(r'\\textbf\{([^}]+)\}', r'<strong>\1</strong>', text)

    # Convert superscripts $^{...}$
    text = re.sub(r'\$\^?\{([^}]+)\}\$', r'<sup>\1</sup>', text)

    # Handle escaped characters
    text = text.replace(r'\&', '&amp;')
    text = text.replace(r'\_', '_')
    text = text.replace(r'\%', '%')
    text = text.replace(r'\$', '$')

    # Remove other LaTeX commands (basic)
    text = re.sub(r'\\[a-zA-Z]+', '', text)

    return text.strip()


def markdown_table_to_html(markdown_table: str) -> str:
    """
    Convert standard Markdown table to HTML.

    Args:
        markdown_table: Markdown table string (| col1 | col2 |)

    Returns:
        HTML table string
    """
    lines = [line.strip() for line in markdown_table.strip().split('\n') if line.strip()]

    if len(lines) < 2:
        return "<p>Invalid markdown table</p>"

    html_parts = ['<table border="1" style="border-collapse: collapse; width: 100%;">']

    for i, line in enumerate(lines):
        # Skip separator line (|---|---|)
        if all(c in '|-: ' for c in line):
            continue

        # Split by | and remove empty first/last elements
        cells = [cell.strip() for cell in line.split('|')]
        cells = [c for c in cells if c]  # Remove empty strings

        # First line is header
        if i == 0:
            html_parts.append('  <thead><tr>')
            for cell in cells:
                html_parts.append(f'    <th>{cell}</th>')
            html_parts.append('  </tr></thead>')
            html_parts.append('  <tbody>')
        else:
            html_parts.append('  <tr>')
            for cell in cells:
                html_parts.append(f'    <td>{cell}</td>')
            html_parts.append('  </tr>')

    html_parts.append('  </tbody>')
    html_parts.append('</table>')

    return '\n'.join(html_parts)


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
    # Remove opening ```markdown or ```html
    content = re.sub(r'^```(?:markdown|html)\s*\n', '', content.strip())

    # Remove closing ```
    content = re.sub(r'\n```\s*$', '', content)

    return content.strip()


def convert_markdown_tables_to_html(markdown_content: str) -> str:
    """
    Automatically convert all tables (LaTeX and Markdown) to HTML within markdown.

    Args:
        markdown_content: Markdown string with LaTeX/Markdown tables

    Returns:
        Modified markdown with HTML tables
    """
    modified_content = markdown_content

    # 1. Convert LaTeX tables to HTML
    latex_tables = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', markdown_content, re.DOTALL)

    for latex_table in latex_tables:
        html_table = latex_table_to_html(latex_table)
        modified_content = modified_content.replace(latex_table, html_table)

    # 2. Convert Markdown tables to HTML
    # Extract markdown tables with their position
    lines = modified_content.split('\n')
    result_lines = []
    current_table_lines = []
    in_table = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('|') and '|' in stripped[1:]:
            current_table_lines.append(line)
            in_table = True
        else:
            if in_table and current_table_lines:
                # End of table - convert it
                md_table = '\n'.join(current_table_lines)
                html_table = markdown_table_to_html(md_table)
                result_lines.append(html_table)
                current_table_lines = []
                in_table = False

            result_lines.append(line)

    # Handle last table if file ends with table
    if current_table_lines:
        md_table = '\n'.join(current_table_lines)
        html_table = markdown_table_to_html(md_table)
        result_lines.append(html_table)

    return '\n'.join(result_lines)


def parse_markdown_bboxes(markdown_content: str) -> list[dict]:
    """
    Parse markdown and extract coordinate comments.

    Format: <!-- Image (x1, y1, x2, y2) --> or <!-- Table (x1, y1, x2, y2) -->

    Args:
        markdown_content: Markdown string with coordinate comments

    Returns:
        List of dicts with element info: {type, bbox}
    """
    pattern = r"<!-- (Image|Table) \(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\) -->"
    matches = re.findall(pattern, markdown_content)

    elements = []
    for match in matches:
        element_type, x1, y1, x2, y2 = match
        elements.append({
            'type': element_type,
            'bbox': (int(x1), int(y1), int(x2), int(y2))
        })

    return elements


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
        'Table': 'red'
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


def main(image_path: str = None):
    """
    Main test function - tests Markdown format with table conversion.

    Args:
        image_path: Optional path to specific image. If None, uses first image in input/
    """
    print("=" * 80)
    print("QwenVL Document Parsing Test - via OpenRouter")
    print("Testing 'qwenvl markdown' format")
    print("=" * 80)

    # Clear output folder
    output_dir = Path("output")
    if output_dir.exists():
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

        print("\nCalling API with 'qwenvl markdown' prompt...")
        markdown_output_raw = call_qwen_html_api(test_image, prompt="qwenvl markdown")

        print("\n" + "-" * 80)
        print("Markdown Response Preview:")
        print("-" * 80)
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

        # Save original Markdown
        output_md = Path("output") / f"{input_name}_qwen_markdown_output.md"
        output_md.write_text(markdown_output, encoding='utf-8')
        print(f"  Saved clean markdown to: {output_md}")

        # Auto-convert tables in markdown to HTML
        print("\n" + "-" * 80)
        print("Auto-Converting Tables in Markdown:")
        print("-" * 80)

        markdown_with_html_tables = convert_markdown_tables_to_html(markdown_output)

        # Count how many tables were converted
        html_table_count = markdown_with_html_tables.count('<table')
        print(f"  Converted {html_table_count} table(s) to HTML within markdown")

        # Save converted markdown
        output_md_converted = Path("output") / f"{input_name}_qwen_markdown_with_html_tables.md"
        output_md_converted.write_text(markdown_with_html_tables, encoding='utf-8')
        print(f"  Saved converted markdown to: {output_md_converted}")

        # Show preview
        if html_table_count > 0:
            # Find first table in converted output
            table_start = markdown_with_html_tables.find('<table')
            table_end = markdown_with_html_tables.find('</table>', table_start) + 8
            if table_start != -1:
                print(f"\n  Preview of first converted table:")
                print(f"  {markdown_with_html_tables[table_start:table_end][:300]}...")

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

        # Convert tables to HTML (both LaTeX and Markdown formats)
        print("\n" + "-" * 80)
        print("Converting Tables to HTML:")
        print("-" * 80)

        # Extract LaTeX tables
        latex_tables = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', markdown_output, re.DOTALL)

        # Extract Markdown tables (lines starting with |)
        markdown_tables = []
        lines = markdown_output.split('\n')
        current_table = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('|') and '|' in stripped[1:]:
                current_table.append(line)
                in_table = True
            elif in_table and current_table:
                # End of table
                markdown_tables.append('\n'.join(current_table))
                current_table = []
                in_table = False

        # Catch last table if file ends with table
        if current_table:
            markdown_tables.append('\n'.join(current_table))

        total_tables = len(latex_tables) + len(markdown_tables)

        if total_tables > 0:
            print(f"\nFound {total_tables} table(s):")
            print(f"  - LaTeX tables: {len(latex_tables)}")
            print(f"  - Markdown tables: {len(markdown_tables)}")

            html_tables_output = ['<html>\n<head>\n<style>']
            html_tables_output.append('table { border-collapse: collapse; width: 100%; margin: 20px 0; }')
            html_tables_output.append('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }')
            html_tables_output.append('th { background-color: #4CAF50; color: white; font-weight: bold; }')
            html_tables_output.append('tr:nth-child(even) { background-color: #f2f2f2; }')
            html_tables_output.append('</style>\n</head>\n<body>')
            html_tables_output.append('<h1>Converted Tables from QwenVL Markdown</h1>')

            table_num = 1

            # Convert LaTeX tables
            for latex_table in latex_tables:
                print(f"\n  Table {table_num} (LaTeX):")
                print(f"    Source length: {len(latex_table)} chars")

                html_table = latex_table_to_html(latex_table)
                print(f"    HTML length: {len(html_table)} chars")

                html_tables_output.append(f'<h2>Table {table_num} (LaTeX Format)</h2>')
                html_tables_output.append(html_table)
                table_num += 1

            # Convert Markdown tables
            for md_table in markdown_tables:
                print(f"\n  Table {table_num} (Markdown):")
                print(f"    Source length: {len(md_table)} chars")

                html_table = markdown_table_to_html(md_table)
                print(f"    HTML length: {len(html_table)} chars")

                html_tables_output.append(f'<h2>Table {table_num} (Markdown Format)</h2>')
                html_tables_output.append(html_table)
                table_num += 1

            html_tables_output.append('</body>\n</html>')

            # Save HTML tables
            output_html_tables = Path("output") / f"{input_name}_qwen_tables_converted.html"
            output_html_tables.write_text('\n'.join(html_tables_output), encoding='utf-8')
            print(f"\n  Converted tables saved to: {output_html_tables}")

            # Show preview of first table
            if latex_tables:
                print("\n  First table preview:")
                first_html = latex_table_to_html(latex_tables[0])
                print(f"  {first_html[:500]}...")
            elif markdown_tables:
                print("\n  First table preview:")
                first_html = markdown_table_to_html(markdown_tables[0])
                print(f"  {first_html[:500]}...")
        else:
            print("\n  No tables found in markdown output")

        # Summary
        print("\n" + "=" * 80)
        print("TEST COMPLETE!")
        print("=" * 80)
        print("\nFormat: 'qwenvl markdown' with table/image coordinates")
        print("\nOutput Files:")
        print(f"  - {output_md} (markdown with coordinates)")
        print(f"  - {output_md_converted} (auto-converted tables)")
        if total_tables > 0:
            print(f"  - {output_html_tables} (standalone tables)")

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
