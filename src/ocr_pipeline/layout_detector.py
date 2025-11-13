"""Layout detection using Qwen3-VL via OpenRouter API"""

import os
import base64
import json
from typing import Dict, List, Union, Literal
from pathlib import Path
from PIL import Image
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv


class LayoutDetector:
    """
    Detects document layout elements and bounding boxes using Qwen3-VL.

    Supports two output formats:
    - html: QwenVL HTML format with data-bbox attributes
    - json: Structured JSON with bbox coordinates
    """

    def __init__(self, api_key: str = None):
        """
        Initialize LayoutDetector with OpenRouter API.

        Args:
            api_key: OpenRouter API key. If None, loads from OPENROUTER_API_KEY env var.
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Provide it via api_key parameter or OPENROUTER_API_KEY env variable."
            )

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        # Use Qwen3-VL-30B-A3B-Thinking for better instruction following and complex layout handling
        self.model = "qwen/qwen3-vl-30b-a3b-thinking"

    def detect_layout(
        self,
        image_input: Union[str, Path, Image.Image],
        output_format: Literal["html", "json"] = "html"
    ) -> Dict:
        """
        Detect document layout elements with bounding boxes.

        Args:
            image_input: Path to image file or PIL Image object
            output_format: Output format - "html" or "json"

        Returns:
            Dictionary containing:
            - format: Output format used
            - raw_output: Raw model response
            - elements: List of detected elements with bboxes
            - image_dimensions: Original image dimensions (width, height)
        """
        # Load and encode image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("image_input must be a file path or PIL Image object")

        # Store original dimensions
        original_width, original_height = image.size

        # Convert image to base64
        base64_image = self._encode_image_to_base64(image)

        # Generate appropriate prompt
        if output_format == "html":
            prompt = """Parse this document to QwenVL HTML format with bounding boxes.
Detect ALL layout elements including:
- Printed text (headers, paragraphs, tables)
- Handwritten text and annotations
- Forms with filled-in handwritten entries

IMPORTANT: Focus ONLY on detecting element types and boundaries. Do NOT attempt text extraction.
Include data-bbox attributes for every element."""
        else:  # json
            prompt = """Analyze this document and detect ALL layout elements including text, visual elements, and handwritten content.

IMPORTANT - Your task is to detect component TYPES and BOUNDARIES only. Do NOT extract text content.

Distinguish between different content types:
- PRINTED text: Uniform font, clean edges, machine-printed (use types: header, paragraph, table)
- HANDWRITTEN text: Irregular, pen/pencil strokes, human-written annotations (use type: handwritten)
- SIGNATURES: Cursive/scrawled handwritten signatures, typically at bottom of documents/checks (use type: signature)
- VISUAL elements: Charts, graphs, diagrams, infographics (use specific types: chart, graph, diagram, infographic, or figure)

Return a JSON array with bounding boxes for:
- Headers (h1, h2, h3) - printed headings
- Paragraphs - printed body text
- Tables - printed table structures with cells
- Handwritten - ONLY hand-written annotations, notes, or filled-in form entries (NOT signatures)
- Signature - Cursive signatures at bottom of documents, checks, contracts (use type: "signature")
- Check/Cheque - Entire bank check/cheque region including all fields (use type: "check" or "cheque")
- Charts/Graphs - bar charts, pie charts, donut charts, line graphs (use type: "chart" or "graph")
- Diagrams - flowcharts, organizational diagrams, technical diagrams (use type: "diagram")
- Infographics - visual data representations, maps with data overlays (use type: "infographic")
- Images/figures - photos, illustrations, decorative images (use type: "image" or "figure")
- Lists
- Any other text blocks

For visual elements, try to be specific:
- If it contains numeric data and visualizations → "chart" or "graph"
- If it's a structured visual diagram → "diagram"
- If it combines text, data, and visuals → "infographic"
- Otherwise → "figure" or "image"

Only mark something as "handwritten" if it is clearly written by hand with pen/pencil, not printed.

Focus ONLY on detecting accurate bounding boxes and element types. Do NOT attempt text extraction.

Format: [{"type": "element_type", "bbox": [x1, y1, x2, y2]}]
Return ONLY valid JSON, no additional text."""

        # Call OpenRouter API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            raw_output = response.choices[0].message.content

            # Parse output based on format
            if output_format == "html":
                elements = self._parse_html_output(raw_output, original_width, original_height)
            else:
                elements = self._parse_json_output(raw_output, original_width, original_height)

            return {
                "format": output_format,
                "raw_output": raw_output,
                "elements": elements,
                "image_dimensions": {"width": original_width, "height": original_height}
            }

        except Exception as e:
            raise RuntimeError(f"Error calling OpenRouter API: {e}")

    def detect_layout_multipass(self, image_input: Union[str, Path, Image.Image]) -> dict:
        """
        Two-pass layout detection for complex images with multiple forms/checks.

        Pass 1: Detect high-level regions (forms, checks, documents)
        Pass 2: Detect detailed elements within each region

        Args:
            image_input: Path to image file or PIL Image object

        Returns:
            Combined layout result in same format as detect_layout()
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("image_input must be a file path or PIL Image object")

        original_width, original_height = image.size
        base64_image = self._encode_image_to_base64(image)

        # Pass 1: Detect high-level regions only
        print(f"      [Pass 1/2] Detecting high-level regions (forms/checks)...")
        pass1_prompt = """Detect only high-level document regions in this image.

IMPORTANT: Only detect large regions like forms, checks, or document sections. Do NOT detect individual text fields or extract text content.

Return a JSON array with bounding boxes for:
- Check - Individual bank checks or cheques
- Form - Forms, documents, or structured layouts
- Document - Separate document pages or sections

For each region, provide a large bounding box that encompasses the entire form/check/document.
Focus ONLY on detecting region types and boundaries. Do NOT attempt text extraction.

Format: [{"type": "check" or "form" or "document", "bbox": [x1, y1, x2, y2]}]
Return ONLY valid JSON, no additional text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            },
                            {
                                "type": "text",
                                "text": pass1_prompt
                            }
                        ]
                    }
                ]
            )

            raw_output = response.choices[0].message.content
            high_level_regions = self._parse_json_output(raw_output, original_width, original_height)

            print(f"      Detected {len(high_level_regions['elements'])} high-level regions")

        except Exception as e:
            raise RuntimeError(f"Error in Pass 1 detection: {e}")

        # Pass 2: Detect detailed elements within each region
        print(f"      [Pass 2/2] Detecting detailed elements in each region...")
        all_elements = []

        for idx, region in enumerate(high_level_regions['elements'], 1):
            x1, y1, x2, y2 = region['bbox']

            # Crop region
            region_image = image.crop((x1, y1, x2, y2))
            region_width, region_height = region_image.size

            print(f"            Processing region {idx}/{len(high_level_regions['elements'])}...")

            try:
                # Detect detailed elements in this region
                region_layout = self.detect_layout(region_image, output_format="json")

                # Adjust coordinates from relative to absolute
                for element in region_layout['elements']:
                    element['bbox'] = [
                        element['bbox'][0] + x1,
                        element['bbox'][1] + y1,
                        element['bbox'][2] + x1,
                        element['bbox'][3] + y1
                    ]
                    all_elements.append(element)

            except Exception as e:
                print(f"            [WARNING] Failed to process region {idx}: {e}")
                continue

        print(f"      Total elements detected across all regions: {len(all_elements)}")

        return {
            'elements': all_elements,
            'image_size': {'width': original_width, 'height': original_height}
        }

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        import io

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Save to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        # Encode to base64
        return base64.b64encode(buffer.read()).decode('utf-8')

    def _parse_html_output(
        self,
        html_output: str,
        img_width: int,
        img_height: int
    ) -> List[Dict]:
        """
        Parse QwenVL HTML output to extract bounding boxes.

        QwenVL uses normalized coordinates (0-1000 scale).
        We convert them to actual pixel coordinates.

        Args:
            html_output: HTML string from Qwen3-VL
            img_width: Original image width
            img_height: Original image height

        Returns:
            List of element dictionaries with bbox and metadata
        """
        # Extract HTML from markdown code blocks if present
        if '```html' in html_output:
            html_output = html_output.split('```html')[1].split('```')[0].strip()
        elif '```' in html_output:
            html_output = html_output.split('```')[1].split('```')[0].strip()

        soup = BeautifulSoup(html_output, 'html.parser')
        elements = []

        # Find all elements with data-bbox attribute
        for elem in soup.find_all(attrs={"data-bbox": True}):
            try:
                # Parse bbox string: "x1 y1 x2 y2" (space-separated, normalized 0-1000)
                bbox_str = elem['data-bbox']
                coords = [int(x) for x in bbox_str.split()]

                if len(coords) != 4:
                    continue

                # Convert from normalized (0-1000) to actual pixels
                x1 = int((coords[0] / 1000.0) * img_width)
                y1 = int((coords[1] / 1000.0) * img_height)
                x2 = int((coords[2] / 1000.0) * img_width)
                y2 = int((coords[3] / 1000.0) * img_height)

                # Get element type
                element_type = elem.name
                element_class = elem.get('class', [])

                elements.append({
                    'type': element_type,
                    'class': element_class,
                    'bbox': [x1, y1, x2, y2],
                    'bbox_normalized': coords  # Keep original for reference
                })

            except (ValueError, IndexError) as e:
                # Skip malformed bboxes
                continue

        return elements

    def _parse_json_output(
        self,
        json_output: str,
        img_width: int,
        img_height: int
    ) -> List[Dict]:
        """
        Parse JSON output with bounding boxes.

        Args:
            json_output: JSON string from model
            img_width: Original image width
            img_height: Original image height

        Returns:
            List of element dictionaries
        """
        # Extract JSON from markdown code blocks if present
        if '```json' in json_output:
            json_output = json_output.split('```json')[1].split('```')[0].strip()
        elif '```' in json_output:
            json_output = json_output.split('```')[1].split('```')[0].strip()

        try:
            data = json.loads(json_output)

            # Normalize format if needed
            if isinstance(data, dict) and 'elements' in data:
                data = data['elements']

            # Process each element
            elements = []
            for item in data:
                bbox = item.get('bbox', item.get('bbox_2d', []))

                if len(bbox) != 4:
                    continue

                # Assume coordinates might be normalized or pixel values
                # If all values are <= 1000, treat as normalized
                if all(coord <= 1000 for coord in bbox):
                    # Normalized coordinates
                    x1 = int((bbox[0] / 1000.0) * img_width)
                    y1 = int((bbox[1] / 1000.0) * img_height)
                    x2 = int((bbox[2] / 1000.0) * img_width)
                    y2 = int((bbox[3] / 1000.0) * img_height)
                    bbox_normalized = bbox
                else:
                    # Pixel coordinates
                    x1, y1, x2, y2 = bbox
                    bbox_normalized = [
                        int((x1 / img_width) * 1000),
                        int((y1 / img_height) * 1000),
                        int((x2 / img_width) * 1000),
                        int((y2 / img_height) * 1000)
                    ]

                elements.append({
                    'type': item.get('type', item.get('label', 'unknown')),
                    'bbox': [x1, y1, x2, y2],
                    'bbox_normalized': bbox_normalized
                })

            return elements

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON output: {e}\nOutput: {json_output[:500]}")
