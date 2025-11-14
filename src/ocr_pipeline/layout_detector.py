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
        Automatically uses tiling for long documents (> 2560px height).

        Args:
            image_input: Path to image file or PIL Image object

        Returns:
            Dictionary containing:
            - raw_output: Raw model response
            - elements: List of detected elements with bboxes
            - image_dimensions: Original image dimensions (width, height)
            - tiling_used: Boolean indicating if tiling was used
            - num_tiles: Number of tiles (if tiling was used)
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("image_input must be a file path or PIL Image object")

        # Store original dimensions
        original_width, original_height = image.size

        # Optimize image for Qwen3-VL
        optimized_image = self._optimize_image_for_qwen(image)
        opt_width, opt_height = optimized_image.size

        # Check if tiling is needed (height > 1600px for better attention coverage)
        # Research shows vision models lose attention on tall images, so use conservative threshold
        if opt_height > 1600:
            print(f"[INFO] Document is tall ({opt_height}px), using tiling strategy")
            return self._detect_layout_with_tiling(optimized_image, original_width, original_height)
        else:
            # Single image detection
            return self._detect_single_image(optimized_image, original_width, original_height)

    def _detect_layout_with_tiling(
        self,
        optimized_image: Image.Image,
        original_width: int,
        original_height: int
    ) -> Dict:
        """
        Detect layout using tiling strategy for long documents.

        Args:
            optimized_image: Optimized PIL Image
            original_width: Original image width
            original_height: Original image height

        Returns:
            Dictionary with detection results
        """
        opt_width, opt_height = optimized_image.size

        # Split into tiles
        tiles = self._tile_long_document(optimized_image)

        # Detect on each tile
        all_elements = []
        for tile_data in tiles:
            tile_result = self._detect_single_tile(tile_data["image"])

            # Adjust coordinates by offset
            for element in tile_result["elements"]:
                element["bbox"][1] += tile_data["offset_y"]  # y1
                element["bbox"][3] += tile_data["offset_y"]  # y2

            all_elements.extend(tile_result["elements"])

        # Deduplicate overlapping detections
        all_elements = self._deduplicate_boxes(all_elements)

        # Scale back to original dimensions
        scale_x = original_width / opt_width
        scale_y = original_height / opt_height

        for element in all_elements:
            element["bbox"] = [
                int(element["bbox"][0] * scale_x),
                int(element["bbox"][1] * scale_y),
                int(element["bbox"][2] * scale_x),
                int(element["bbox"][3] * scale_y)
            ]

        return {
            "raw_output": f"Tiled detection: {len(tiles)} tiles, {len(all_elements)} elements",
            "elements": all_elements,
            "image_dimensions": {"width": original_width, "height": original_height},
            "tiling_used": True,
            "num_tiles": len(tiles)
        }

    def _detect_single_image(
        self,
        optimized_image: Image.Image,
        original_width: int,
        original_height: int
    ) -> Dict:
        """
        Detect layout on a single image (no tiling).

        Args:
            optimized_image: Optimized PIL Image
            original_width: Original image width
            original_height: Original image height

        Returns:
            Dictionary with detection results
        """
        opt_width, opt_height = optimized_image.size

        # Detect on single image
        result = self._detect_single_tile(optimized_image)

        # Scale coordinates back to original dimensions
        scale_x = original_width / opt_width
        scale_y = original_height / opt_height

        for element in result["elements"]:
            element["bbox"] = [
                int(element["bbox"][0] * scale_x),
                int(element["bbox"][1] * scale_y),
                int(element["bbox"][2] * scale_x),
                int(element["bbox"][3] * scale_y)
            ]

        return {
            "raw_output": result["raw_output"],
            "elements": result["elements"],
            "image_dimensions": {"width": original_width, "height": original_height},
            "tiling_used": False
        }

    def _detect_single_tile(self, image: Image.Image) -> Dict:
        """
        Detect layout on a single tile/image.

        Args:
            image: PIL Image

        Returns:
            Dictionary with raw output and elements
        """
        # Store tile dimensions
        tile_width, tile_height = image.size

        # Convert image to base64
        base64_image = self._encode_image_to_base64(image)

        # SIMPLIFIED PROMPT: Segment-based detection with accuracy focus and grounding
        prompt = """Find all content regions in this document with grounding.

DOCUMENT CHARACTERISTICS:
- This may be a LONG document with content from top to bottom
- Pay EQUAL ATTENTION to elements at the BOTTOM as well as the TOP
- Scan the ENTIRE vertical space systematically

A SEGMENT is any distinct content block, including:

TEXT CONTENT:
- Paragraphs (body text) - EACH paragraph is a separate segment
- Headings and subheadings - Each heading is its own segment
- Captions and labels
- Footnotes and page numbers
- List items - Each bullet/numbered item can be a segment or group related items

VISUAL CONTENT:
- Images and photos
- Logos and icons
- Diagrams and illustrations
- Infographics

STRUCTURED DATA:
- Tables (entire table as one segment)
- Charts and graphs
- Data visualizations

HANDWRITTEN CONTENT:
- Handwritten text
- Signatures
- Annotations and markups

FORM ELEMENTS:
- Checkboxes and radio buttons
- Form fields
- Stamps and seals

MATHEMATICAL CONTENT:
- Equations and formulas

LAYOUT ELEMENTS:
- Headers and footers
- Sidebars

CRITICAL RULES:
1. SYSTEMATIC SCAN: Start from the top and scan downward to the very bottom
2. COMPLETE COVERAGE: Detect ALL content regions - DO NOT stop scanning halfway
3. BOTTOM ELEMENTS: Elements near the bottom are just as important as top elements
4. SEPARATE PARAGRAPHS: Each distinct paragraph/text block must be its own segment
5. NO OVERLAPPING: Bounding boxes must NOT overlap with each other
6. TIGHT BOUNDARIES: Draw precise boxes around each content region
   - x1, y1 = top-left corner (start of content)
   - x2, y2 = bottom-right corner (end of content)
7. VISUAL SEPARATION: If there's whitespace between text blocks, they are separate segments
8. COORDINATE SYSTEM: Return coordinates normalized to 1000x1000 reference frame

IMPORTANT: All segments must have type "segment" - do not attempt to classify them.

Format: Return with grounding in JSON format:
[{"type": "segment", "bbox": [x1, y1, x2, y2]}]
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
            elements = self._parse_json_output(raw_output, tile_width, tile_height)

            return {
                "raw_output": raw_output,
                "elements": elements
            }

        except Exception as e:
            raise RuntimeError(f"Error calling OpenRouter API: {e}")

    def _optimize_image_for_qwen(self, image: Image.Image) -> Image.Image:
        """
        Resize image to optimal range for Qwen3-VL detection.

        Target: 1000-1200px width, height proportional
        Dimensions must be multiples of 28 (Qwen ViT requirement)
        Note: Heights > 1600px will be tiled for better attention coverage

        Args:
            image: PIL Image

        Returns:
            Optimized PIL Image
        """
        width, height = image.size
        aspect_ratio = height / width

        # Check aspect ratio limit (Qwen requires < 200)
        if max(height, width) / min(height, width) >= 200:
            raise ValueError(f"Aspect ratio too extreme: {max(height, width) / min(height, width):.2f}")

        # Target width: 1100px (optimal for document OCR per Qwen docs)
        target_width = 1100

        # Calculate target height
        target_height = int(target_width * aspect_ratio)

        # If still too wide/tall, scale proportionally (max 2560px for model limits)
        if target_height > 2560 or target_width > 2560:
            scale_factor = min(2560 / target_width, 2560 / target_height)
            target_width = int(target_width * scale_factor)
            target_height = int(target_height * scale_factor)

        # Ensure dimensions are multiples of 28 (Qwen requirement)
        target_width = (target_width // 28) * 28
        target_height = (target_height // 28) * 28

        # Only resize if dimensions changed significantly (>10%)
        if abs(width - target_width) / width > 0.1 or abs(height - target_height) / height > 0.1:
            print(f"[INFO] Optimizing image: {width}x{height} -> {target_width}x{target_height}")
            resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            return resized

        return image

    def _tile_long_document(
        self,
        image: Image.Image,
        tile_height: int = 1400,
        overlap: int = 200
    ) -> List[Dict]:
        """
        Split long document into overlapping tiles for better detection.

        Args:
            image: PIL Image
            tile_height: Height of each tile (should be ~1400px for optimal attention)
            overlap: Overlap between tiles to avoid missing boundary elements

        Returns:
            List of tiles with metadata for reassembly
        """
        width, height = image.size

        # If image is short enough, return as single tile
        if height <= 1600:
            return [{
                "image": image,
                "offset_y": 0,
                "tile_index": 0,
                "overlap_top": 0
            }]

        tiles = []
        current_y = 0
        tile_index = 0

        while current_y < height:
            # Calculate tile boundaries
            y_start = max(0, current_y - overlap if current_y > 0 else 0)
            y_end = min(height, current_y + tile_height)

            # Crop tile
            tile = image.crop((0, y_start, width, y_end))

            tiles.append({
                "image": tile,
                "offset_y": y_start,
                "tile_index": tile_index,
                "overlap_top": overlap if current_y > 0 else 0
            })

            tile_index += 1
            current_y += tile_height - overlap

            if y_end >= height:
                break

        print(f"[INFO] Split document into {len(tiles)} tiles ({height}px height)")
        return tiles

    def _deduplicate_boxes(self, elements: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Remove duplicate bounding boxes in overlap zones.

        Args:
            elements: List of detected elements with bboxes
            iou_threshold: IoU threshold for considering boxes duplicates

        Returns:
            Deduplicated list of elements
        """
        if len(elements) <= 1:
            return elements

        def calculate_iou(box1, box2):
            """Calculate Intersection over Union of two boxes."""
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2

            # Intersection
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
                return 0.0

            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

            # Union
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = box1_area + box2_area - inter_area

            return inter_area / union_area if union_area > 0 else 0.0

        # Sort by y-coordinate to process top-to-bottom
        sorted_elements = sorted(elements, key=lambda e: e['bbox'][1])

        keep = []
        for i, elem in enumerate(sorted_elements):
            is_duplicate = False
            for kept_elem in keep:
                iou = calculate_iou(elem['bbox'], kept_elem['bbox'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep.append(elem)

        print(f"[INFO] Deduplication: {len(elements)} -> {len(keep)} elements")
        return keep

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


