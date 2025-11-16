"""
QwenVL-based Document Extraction Module.

This module provides a simplified single-stage OCR pipeline using QwenVL
via OpenRouter API. It replaces the complex multi-stage pipeline with a
single API call that extracts markdown with coordinate annotations.
"""

import os
import base64
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from io import BytesIO

import requests
from PIL import Image, ImageFilter, ImageEnhance
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Import table HTML converter (conditional to avoid circular imports)
try:
    from .table_html_converter import convert_table_to_html
except ImportError:
    # Fallback if module not available
    convert_table_to_html = None


def smart_resize(
    height: int,
    width: int,
    min_pixels: int = 512 * 32 * 32,
    max_pixels: int = 2048 * 32 * 32,
    factor: int = 32
) -> Tuple[int, int]:
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


def analyze_image_quality(image: Image.Image) -> Dict:
    """
    Analyze image quality metrics to determine if preprocessing is needed.

    Args:
        image: PIL Image object

    Returns:
        Dict with quality metrics:
        - sharpness: Laplacian variance (higher = sharper, >100 is good)
        - contrast: Standard deviation of histogram (higher = better, >50 is good)
        - resolution: Total pixels
        - width, height: Dimensions
    """
    # Convert to grayscale for analysis
    gray = image.convert('L')
    gray_array = np.array(gray)

    # Sharpness: Laplacian variance
    # Higher values = sharper image
    # Typical ranges: <50 = blurry, 50-100 = moderate, >100 = sharp
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    filtered = np.abs(np.convolve(gray_array.flatten(), laplacian.flatten(), mode='same'))
    sharpness = filtered.var()

    # Contrast: Standard deviation of pixel values
    # Higher values = better contrast
    # Typical ranges: <30 = low contrast, 30-60 = moderate, >60 = high contrast
    contrast = gray_array.std()

    # Resolution
    width, height = image.size
    resolution = width * height

    return {
        'sharpness': float(sharpness),
        'contrast': float(contrast),
        'resolution': resolution,
        'width': width,
        'height': height,
        'needs_preprocessing': sharpness < 100 or contrast < 50 or resolution < 1000000
    }


def preprocess_image(
    image: Image.Image,
    enhance_contrast: bool = True,
    sharpen: bool = True,
    upscale: bool = True,
    max_pixels: int = 4608 * 32 * 32  # QwenVL's max_pixels for markdown (4.7M)
) -> Image.Image:
    """
    Preprocess image to improve OCR quality.

    Applies conservative enhancements that improve readability without
    over-processing or creating artifacts.

    Args:
        image: PIL Image object
        enhance_contrast: Apply CLAHE-style contrast enhancement
        sharpen: Apply unsharp mask sharpening
        upscale: Upscale if resolution is too low (respects max_pixels)
        max_pixels: Maximum total pixels (default: 2048*32*32 for QwenVL)

    Returns:
        Preprocessed PIL Image
    """
    result = image.copy()

    # Step 1: Upscale if resolution is too low, but respect max_pixels
    if upscale:
        width, height = result.size
        current_pixels = width * height

        # Only upscale if current resolution is low AND won't exceed max_pixels
        # Target: use 70% of max_pixels to leave headroom for QwenVL processing
        target_pixels = int(max_pixels * 0.7)  # 1,468,006 pixels

        if current_pixels < target_pixels:
            scale_factor = (target_pixels / current_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Double-check we don't exceed max_pixels
            if new_width * new_height <= max_pixels:
                # Use Lanczos resampling (high quality)
                result = result.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"  Upscaled: {width}x{height} -> {new_width}x{new_height} "
                      f"({current_pixels} -> {new_width*new_height} pixels)")
            else:
                print(f"  Skipped upscaling: would exceed max_pixels ({max_pixels})")

    # Step 2: Enhance contrast (CLAHE approximation using Pillow)
    if enhance_contrast:
        # Convert to LAB color space equivalent using RGB channels
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(1.3)  # 30% contrast boost
        print(f"  Applied contrast enhancement")

    # Step 3: Sharpen to enhance text edges
    if sharpen:
        # Unsharp mask: enhances edges without creating artifacts
        result = result.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
        print(f"  Applied sharpening")

    return result


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


def encode_pil_image_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode PIL Image to base64 string.

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Base64 encoded string
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_qwen_markdown_api(
    image_input,  # Can be str (file path) or PIL Image
    prompt: str = "qwenvl markdown",
    min_pixels: int = 512 * 32 * 32,
    max_pixels: int = 2048 * 32 * 32,
    include_usage: bool = False
):
    """
    Call OpenRouter API with Qwen3-VL model to get Markdown output.

    Args:
        image_input: Either a file path (str) or PIL Image object
        prompt: Prompt to send (default: "qwenvl markdown")
        min_pixels: Minimum pixels for image resize
        max_pixels: Maximum pixels for image resize
        include_usage: Whether to include usage/cost data in response

    Returns:
        If include_usage=False: Markdown response string with coordinate annotations
        If include_usage=True: Tuple of (markdown_string, usage_dict)
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    # Handle both file path and PIL Image
    if isinstance(image_input, str):
        # Load and encode image from file
        base64_image = encode_image_base64(image_input)
        with Image.open(image_input) as img:
            width, height = img.size
        image_format = Path(image_input).suffix.lower().replace(".", "")
        if image_format == "jpg":
            image_format = "jpeg"
    elif isinstance(image_input, Image.Image):
        # Encode PIL Image
        base64_image = encode_pil_image_base64(image_input)
        width, height = image_input.size
        image_format = "png"  # Default to PNG for PIL images
    else:
        raise ValueError("image_input must be either a file path (str) or PIL Image object")

    # Get smart resize dimensions
    input_height, input_width = smart_resize(
        height, width, min_pixels=min_pixels, max_pixels=max_pixels
    )

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

    # Add usage tracking if requested
    if include_usage:
        payload["usage"] = {"include": True}

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    # Return usage data if requested
    if include_usage:
        return content, result.get("usage", {})
    else:
        return content


def clean_markdown_wrapper(content: str) -> str:
    """
    Remove code fence wrappers from markdown content.

    QwenVL often returns: ```markdown\\n...\\n```
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


def parse_markdown_bboxes(markdown_content: str) -> List[Dict]:
    """
    Parse markdown and extract coordinate comments.

    Format: <!-- Image (x1, y1, x2, y2) --> or <!-- Table (x1, y1, x2, y2) -->

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
        match = re.match(
            r'<!-- (Image|Table|Chart|Signature|Handwritten) \((\d+), (\d+), (\d+), (\d+)\) -->',
            line
        )

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


def convert_bbox_to_pixels(
    bbox: Tuple[int, int, int, int],
    image_width: int,
    image_height: int
) -> Tuple[int, int, int, int]:
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


def replace_latex_in_markdown(
    markdown: str,
    old_tables: List[str],
    new_tables: List[str]
) -> str:
    """
    Replace LaTeX tables in markdown in order.

    Args:
        markdown: Markdown content with LaTeX tables
        old_tables: List of original LaTeX table strings
        new_tables: List of corrected LaTeX table strings

    Returns:
        Updated markdown with corrected tables
    """
    result = markdown
    for old, new in zip(old_tables, new_tables):
        # Replace first occurrence only (to maintain order)
        result = result.replace(old, new, 1)
    return result


def extract_images_from_markdown(
    image_path: str,
    markdown_content: str,
    extract_types: Optional[set] = None
) -> Tuple[str, List[Dict]]:
    """
    Extract images from markdown coordinate annotations and embed them inline.

    Args:
        image_path: Path to the original image
        markdown_content: Markdown string with coordinate comments
        extract_types: Set of element types to extract (default: Image, Table, Chart, Signature, Handwritten)

    Returns:
        Tuple of (markdown_with_inline_images, extracted_images_array)
        - markdown_with_inline_images: Markdown with base64 images embedded
        - extracted_images_array: List of dicts with {type, base64, bbox}
    """
    if extract_types is None:
        extract_types = {'Image', 'Table', 'Chart', 'Signature', 'Handwritten'}

    # Parse elements
    elements = parse_markdown_bboxes(markdown_content)

    # Load original image
    with Image.open(image_path) as img:
        width, height = img.size

        # Extract images
        extracted_images = []
        for elem in elements:
            if elem['type'] not in extract_types:
                continue

            bbox = elem['bbox']

            # Convert 0-1000 coordinates to pixels
            px1, py1, px2, py2 = convert_bbox_to_pixels(bbox, width, height)

            # Crop the region
            cropped = img.crop((px1, py1, px2, py2))

            # Encode to base64
            base64_image = encode_pil_image_base64(cropped, format="PNG")

            extracted_images.append({
                'type': elem['type'],
                'base64': base64_image,
                'bbox': bbox
            })

    # Create inline markdown by replacing coordinate comments with images
    markdown_with_images = markdown_content
    for i, img_data in enumerate(extracted_images):
        elem_type = img_data['type']
        bbox = img_data['bbox']
        base64_data = img_data['base64']

        # Find the coordinate comment
        pattern = f"<!-- {elem_type} ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}) -->"

        # Replace with image + comment
        replacement = f"{pattern}\n![{elem_type}](data:image/png;base64,{base64_data})"
        markdown_with_images = markdown_with_images.replace(pattern, replacement)

    return markdown_with_images, extracted_images


def extract_document(
    image_path: str,
    min_pixels: int = 512 * 32 * 32,
    max_pixels: int = 4608 * 32 * 32,  # QwenVL markdown format supports up to 4608*32*32
    include_images: bool = True,
    include_usage: bool = False,
    convert_tables_to_html: bool = False,
    preprocess: bool = True
) -> Dict:
    """
    Extract document content using QwenVL single-stage pipeline.

    Main entry point for document extraction. Returns markdown with optional
    inline base64 images and a separate array of extracted images.

    Args:
        image_path: Path to the image file
        min_pixels: Minimum pixels for image resize
        max_pixels: Maximum pixels for image resize
        include_images: Whether to extract and embed images
        include_usage: Whether to include usage/cost data in response
        convert_tables_to_html: Whether to convert LaTeX tables to HTML using Gemini 2.5 Flash
        preprocess: Whether to apply image preprocessing (contrast enhancement, sharpening,
                    upscaling). Default=True. Improves accuracy by ~4% on low-quality images
                    with minimal overhead (~100-200ms). Recommended to keep enabled.

    Returns:
        Dict with:
        - markdown: Markdown with inline base64 images (if include_images=True)
        - images: List of extracted images [{type, base64, bbox}]
        - elements: List of detected elements with coordinates
        - usage: Usage/cost data (if include_usage=True)
        - quality: Image quality metrics (if preprocess=True)
        - success: Boolean indicating success
        - error: Error message if success=False
    """
    try:
        # Load original image
        original_image = Image.open(image_path)

        # Preprocess if requested
        quality_metrics = None
        image_to_process = original_image

        if preprocess:
            # Analyze image quality
            quality_metrics = analyze_image_quality(original_image)
            print(f"Image quality: sharpness={quality_metrics['sharpness']:.1f}, "
                  f"contrast={quality_metrics['contrast']:.1f}, "
                  f"resolution={quality_metrics['width']}x{quality_metrics['height']}")

            # Apply preprocessing if image quality is poor
            if quality_metrics['needs_preprocessing']:
                print("Preprocessing image...")
                image_to_process = preprocess_image(
                    original_image,
                    enhance_contrast=True,
                    sharpen=True,
                    upscale=True
                )
                quality_metrics['preprocessing_applied'] = True
            else:
                print("Image quality is good, skipping preprocessing")
                quality_metrics['preprocessing_applied'] = False
        # Enhanced prompt for QwenVL
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

CRITICAL TABLE EXTRACTION RULES:
- Pay EXTRA attention to faint, low-contrast, or difficult-to-read text in table cells
- CAREFULLY align each value with its correct column header - count columns meticulously
- If a cell appears empty or blank, output the value as empty (do NOT skip the column)
- For multi-line cells, combine lines with '/' separator (e.g., "104/22")
- Double-check column alignment: header row column count MUST match every data row column count
- Look closely at table borders and gridlines to determine cell boundaries
- If text is partially visible or cut off, transcribe what you can see
- For numbers, pay special attention to distinguishing similar digits (0 vs 8, 1 vs 7, 3 vs 8)
- Preserve exact numeric values - do NOT round or approximate
"""

        # Call QwenVL API (use preprocessed image if available)
        api_result = call_qwen_markdown_api(
            image_to_process,
            prompt=enhanced_prompt,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            include_usage=include_usage
        )

        # Handle both return types (string or tuple)
        if include_usage:
            markdown_raw, usage = api_result
        else:
            markdown_raw = api_result
            usage = {}

        # Clean markdown (remove code fences)
        markdown_clean = clean_markdown_wrapper(markdown_raw)

        # Add alignment tags if model didn't add them
        if '<div align=' not in markdown_clean:
            markdown_clean = add_alignment_to_markdown(markdown_clean)

        # Parse elements
        elements = parse_markdown_bboxes(markdown_clean)

        # Extract images if requested
        if include_images:
            markdown_with_images, extracted_images = extract_images_from_markdown(
                image_path,
                markdown_clean
            )
        else:
            markdown_with_images = markdown_clean
            extracted_images = []

        # Convert tables to HTML if requested
        if convert_tables_to_html and extracted_images:
            markdown_with_images = replace_latex_tables_with_html(
                markdown_with_images,
                extracted_images
            )

        return {
            'success': True,
            'markdown': markdown_with_images,
            'images': extracted_images,
            'elements': elements,
            'usage': usage,
            'quality': quality_metrics,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'markdown': '',
            'images': [],
            'elements': [],
            'usage': {},
            'quality': quality_metrics if 'quality_metrics' in locals() else None,
            'error': str(e)
        }


def replace_latex_tables_with_html(
    markdown: str,
    extracted_images: List[Dict]
) -> str:
    """
    Replace LaTeX tables with HTML versions using Gemini 2.5 Flash.

    This function finds all table elements in the markdown (identified by
    <!-- Table (...) --> markers), extracts their LaTeX content, and converts
    them to clean HTML using the table_html_converter module.

    Args:
        markdown: Markdown string with LaTeX tables
        extracted_images: List of extracted images (includes table images with base64 data)

    Returns:
        Markdown with HTML tables replacing LaTeX tables

    Example:
        Before:
            <!-- Table (100, 200, 800, 600) -->
            \\begin{tabular}{|l|l|}...\\end{tabular}

        After:
            <!-- Table (100, 200, 800, 600) -->
            <table><thead>...</thead><tbody>...</tbody></table>
    """
    if not convert_table_to_html:
        # Table converter not available, return unchanged
        print("Warning: table_html_converter not available, skipping HTML conversion")
        return markdown

    # Find all table elements
    table_images = [img for img in extracted_images if img['type'] == 'Table']

    if not table_images:
        # No tables to convert
        return markdown

    # Process each table
    for table_img in table_images:
        bbox = table_img['bbox']
        base64_data = table_img['base64']

        # Build pattern to find the table marker
        # Escape parentheses for regex
        pattern = f"<!-- Table \\({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}\\) -->"

        # Find position of marker in markdown
        match = re.search(pattern, markdown)
        if not match:
            print(f"Warning: Could not find table marker for bbox {bbox}")
            continue

        # Extract LaTeX content after the marker
        start_pos = match.end()

        # Find LaTeX table: \begin{tabular}...\end{tabular}
        # Use DOTALL flag to match across newlines
        latex_pattern = r'\\begin\{tabular\}.*?\\end\{tabular\}'
        latex_match = re.search(latex_pattern, markdown[start_pos:], re.DOTALL)

        if latex_match:
            latex_content = latex_match.group(0)

            try:
                # Convert to HTML using Gemini 2.5 Flash
                print(f"Converting table at bbox {bbox} to HTML...")
                html_table = convert_table_to_html(base64_data, latex_content)

                # Replace LaTeX with HTML in markdown
                markdown = markdown.replace(latex_content, html_table)
                print(f"Successfully converted table at bbox {bbox}")

            except Exception as e:
                # If conversion fails, keep LaTeX (fallback to original)
                print(f"Warning: Table HTML conversion failed for bbox {bbox}: {e}")
                continue
        else:
            print(f"Warning: Could not find LaTeX content for table at bbox {bbox}")

    return markdown
