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

        # SIMPLIFIED PROMPT: Segment-based detection with accuracy focus
        prompt = """Find all content regions in this document.

Your task: Detect precise bounding boxes for each distinct content block.

CRITICAL - Bounding box accuracy rules:
- Draw tight boundaries around each content region
- x1, y1 = top-left corner (start of content)
- x2, y2 = bottom-right corner (end of content)
- Be pixel-precise - ensure boxes align exactly with content edges
- Include all content within the region, but avoid excess whitespace

IMPORTANT: All elements must have type "segment" - do not classify the content type.

Format: [{"type": "segment", "bbox": [x1, y1, x2, y2]}]
Return ONLY valid JSON."""

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


