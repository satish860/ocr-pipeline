"""
QwenVL Extraction Component.

Handles communication with QwenVL API to extract document content as markdown
with coordinate annotations.
"""

import os
from pathlib import Path
from typing import Dict

import requests
from PIL import Image
from dotenv import load_dotenv

from .utils import (
    smart_resize,
    encode_image_base64,
    encode_pil_image_base64,
    clean_markdown_wrapper,
    add_alignment_to_markdown,
    parse_markdown_bboxes,
    extract_images_from_markdown
)

# Load environment variables
load_dotenv()


class QwenExtractor:
    """
    QwenVL API extraction component.

    Handles communication with QwenVL API to extract document content
    as markdown with coordinate annotations.
    """

    def __init__(
        self,
        min_pixels: int = 512 * 32 * 32,
        max_pixels: int = 4608 * 32 * 32
    ):
        """
        Initialize QwenExtractor.

        Args:
            min_pixels: Minimum pixels for image resize
            max_pixels: Maximum pixels for image resize
        """
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def extract(
        self,
        image_input,  # Can be str (file path) or PIL Image
        include_images: bool = True,
        include_usage: bool = False
    ) -> Dict:
        """
        Extract document content using QwenVL API.

        Args:
            image_input: Either a file path (str) or PIL Image object
            include_images: Whether to extract and embed images inline
            include_usage: Whether to include usage/cost data in response

        Returns:
            Dict with:
            - success: Boolean indicating success
            - markdown: Markdown string with inline base64 images (if include_images=True)
            - images: List of extracted images [{type, base64, bbox}]
            - elements: List of detected elements with coordinates
            - usage: Usage/cost data (if include_usage=True)
            - error: Error message if success=False
        """
        try:
            # Enhanced prompt for QwenVL
            enhanced_prompt = """qwenvl markdown

Convert this document to markdown format with the following requirements:
- Extract ALL text content as markdown in natural reading order (top to bottom, left to right)
- Preserve text alignment using HTML tags:
  - Right-aligned text: <div align="right">text</div>
  - Center-aligned text: <div align="center">text</div>
  - Left-aligned text: no tags needed (default)
- Use <br> tags for line breaks within aligned sections
- Represent all tables in LaTeX format using \\\\begin{tabular} and \\\\end{tabular}
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

            # Call QwenVL API
            api_result = self._call_qwen_api(
                image_input,
                prompt=enhanced_prompt,
                include_usage=include_usage
            )

            if not api_result['success']:
                return api_result

            # Extract markdown and usage
            markdown_raw = api_result['markdown']
            usage = api_result.get('usage', {})

            # Clean markdown (remove code fences)
            markdown_clean = clean_markdown_wrapper(markdown_raw)

            # Add alignment tags if model didn't add them
            if '<div align=' not in markdown_clean:
                markdown_clean = add_alignment_to_markdown(markdown_clean)

            # Parse elements
            elements = parse_markdown_bboxes(markdown_clean)

            # Extract images if requested
            if include_images:
                # Determine image_path for extraction
                if isinstance(image_input, str):
                    image_path = image_input
                else:
                    # PIL Image - need to save temporarily to extract regions
                    # For now, skip image extraction for PIL Images
                    # TODO: Support PIL Image extraction by working directly with Image object
                    return {
                        'success': True,
                        'markdown': markdown_clean,
                        'images': [],
                        'elements': elements,
                        'usage': usage,
                        'error': None
                    }

                markdown_with_images, extracted_images = extract_images_from_markdown(
                    image_path,
                    markdown_clean
                )
            else:
                markdown_with_images = markdown_clean
                extracted_images = []

            return {
                'success': True,
                'markdown': markdown_with_images,
                'images': extracted_images,
                'elements': elements,
                'usage': usage,
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'markdown': '',
                'images': [],
                'elements': [],
                'usage': {},
                'error': str(e)
            }

    def _call_qwen_api(
        self,
        image_input,  # Can be str (file path) or PIL Image
        prompt: str = "qwenvl markdown",
        include_usage: bool = False
    ) -> Dict:
        """
        Internal method to call OpenRouter API with Qwen3-VL model.

        Args:
            image_input: Either a file path (str) or PIL Image object
            prompt: Prompt to send
            include_usage: Whether to include usage/cost data in response

        Returns:
            Dict with:
            - success: Boolean indicating success
            - markdown: Raw markdown response string
            - usage: Usage/cost data (if include_usage=True)
            - error: Error message if success=False
        """
        try:
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
                height, width, min_pixels=self.min_pixels, max_pixels=self.max_pixels
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
                                "min_pixels": self.min_pixels,
                                "max_pixels": self.max_pixels,
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

            return {
                'success': True,
                'markdown': content,
                'usage': result.get("usage", {}) if include_usage else {},
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'markdown': '',
                'usage': {},
                'error': str(e)
            }
