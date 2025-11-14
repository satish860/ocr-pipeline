"""Layout detection using Qwen3-VL via OpenRouter API"""

import os
import base64
import json
from typing import Dict, List, Union
from pathlib import Path
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv


class LayoutDetector:
    """
    Detects document layout elements and bounding boxes using Qwen3-VL.

    Output format: Structured JSON with bbox coordinates
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
        image_input: Union[str, Path, Image.Image]
    ) -> Dict:
        """
        Detect document layout elements with bounding boxes.

        Args:
            image_input: Path to image file or PIL Image object

        Returns:
            Dictionary containing:
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

        # Generate JSON detection prompt - Detect BOTH structure AND content
        prompt = """Analyze this document and detect ALL layout elements - BOTH structural containers AND content inside them.

CRITICAL: Detect BOTH structure AND content as SEPARATE elements:
- A signature box (container) AND the text/signature inside it
- A field label (e.g., "Signature:") AND the field box AND the content
- These are NOT mutually exclusive - return ALL of them

IMPORTANT - Your task is to detect component TYPES and BOUNDARIES only. Do NOT extract text content.

=== STRUCTURAL CONTAINERS (boxes, fields, regions) ===
- signature_box - Rectangular container/field for signatures (often has underline or border)
- box / field - Input fields, rectangular containers, form fields
- checkbox - Small boxes for checkmarks
- table - Table structures with cells and borders
- line / separator - Lines, underlines, borders, dividers

=== TEXT CONTENT (for OCR to extract later) ===
- header - Document titles, section headings, field labels
- paragraph - Body text, descriptions, instructions
- label - Short field identifiers (e.g., "Name:", "Date:", "Signature:")
- handwritten - Hand-written annotations, notes (irregular pen/pencil)
- signature - Actual handwritten signature marks (cursive/scrawled)

=== VISUAL ELEMENTS ===
- chart / graph - Bar charts, pie charts, line graphs, data visualizations
- diagram - Flowcharts, organizational diagrams, technical diagrams
- infographic - Visual data representations, maps with data overlays
- image / figure - Photos, illustrations, decorative images
- check / cheque - Bank check/cheque region

IMPORTANT EXAMPLES:

Example 1 - Signature area with label and box:
If you see: "Signature: _______________" with a handwritten signature on the line
Detect 3 SEPARATE elements:
1. {"type": "header", "bbox": [x1, y1, x2, y2]} - The "Signature:" text
2. {"type": "signature_box", "bbox": [x3, y3, x4, y4]} - The underline/box container
3. {"type": "signature", "bbox": [x5, y5, x6, y6]} - The actual handwritten signature

Example 2 - Form field with label:
If you see: "Name: [___________]" with handwritten text inside
Detect 3 SEPARATE elements:
1. {"type": "label", "bbox": [...]} - The "Name:" text
2. {"type": "box", "bbox": [...]} - The input field box
3. {"type": "handwritten", "bbox": [...]} - The written name inside

Example 3 - Text inside a box:
If a paragraph is inside a bordered box, detect BOTH:
1. {"type": "box", "bbox": [...]} - The container box
2. {"type": "paragraph", "bbox": [...]} - The text inside

RULES:
- Detect containers (boxes, fields) separately from their content (text, signatures)
- Detect labels separately from the fields they describe
- A single visual region can have MULTIPLE elements (container + content + label)
- Return ALL detected elements, even if they overlap

Focus on accurate bounding boxes for ALL elements.

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

            # Parse JSON output
            elements = self._parse_json_output(raw_output, original_width, original_height)

            return {
                "raw_output": raw_output,
                "elements": elements,
                "image_dimensions": {"width": original_width, "height": original_height}
            }

        except Exception as e:
            raise RuntimeError(f"Error calling OpenRouter API: {e}")


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


