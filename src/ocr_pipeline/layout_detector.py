"""Layout detection using Qwen3-VL via OpenRouter API"""

import os
import base64
import json
from typing import Dict, List, Union
from pathlib import Path
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from .image_preprocessor import ImagePreprocessor


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
        image_input: Union[str, Path, Image.Image],
        preprocess: bool = False
    ) -> Dict:
        """
        Detect document layout elements with bounding boxes.

        Args:
            image_input: Path to image file or PIL Image object
            preprocess: Apply image preprocessing (CLAHE, sharpening) for better detection

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

        # Apply preprocessing if requested
        if preprocess:
            preprocessor = ImagePreprocessor()
            image = preprocessor.preprocess_for_ocr(
                image,
                deskew=False,  # Don't deskew for layout detection
                enhance_contrast=False,
                clahe=True,  # Use CLAHE for better contrast
                sharpen=True,  # Sharpen to make lines/text clearer
                denoise=False,  # Skip denoising (slow and not always needed)
                upscale=1.0  # No upscaling (model handles various sizes well)
            )

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
                region_layout = self.detect_layout(region_image)

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

    def _analyze_gaps(self, elements: List[Dict], gap_threshold: int = 100) -> List[tuple]:
        """
        Analyze vertical gaps between detected elements.

        Args:
            elements: List of detected elements with bboxes
            gap_threshold: Minimum gap size (in pixels) to flag as suspicious

        Returns:
            List of gap regions: [(y_start, y_end), ...]
        """
        if len(elements) < 2:
            return []

        # Sort elements by vertical position (y1)
        sorted_elements = sorted(elements, key=lambda e: e['bbox'][1])

        gaps = []
        for i in range(len(sorted_elements) - 1):
            current_y2 = sorted_elements[i]['bbox'][3]  # Bottom of current element
            next_y1 = sorted_elements[i + 1]['bbox'][1]  # Top of next element
            gap_size = next_y1 - current_y2

            if gap_size > gap_threshold:
                gaps.append((current_y2, next_y1, gap_size))

        return gaps

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]

        Returns:
            IoU score (0.0 to 1.0)
        """
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _merge_elements(self, initial_elements: List[Dict], gap_elements: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Merge elements from initial detection and gap detection, removing duplicates.

        Args:
            initial_elements: Elements from first detection pass
            gap_elements: Elements from gap-focused detection
            iou_threshold: IoU threshold for considering elements as duplicates

        Returns:
            Merged list of unique elements
        """
        merged = initial_elements.copy()

        for gap_elem in gap_elements:
            is_duplicate = False
            for init_elem in initial_elements:
                iou = self._calculate_iou(gap_elem['bbox'], init_elem['bbox'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                merged.append(gap_elem)

        # Sort by vertical position
        merged.sort(key=lambda e: e['bbox'][1])

        return merged

    def _detect_in_gaps(
        self,
        image: Image.Image,
        gaps: List[tuple],
        existing_bboxes: List[List[int]],
        img_width: int,
        img_height: int
    ) -> List[Dict]:
        """
        Perform targeted detection in identified gaps using self-critique approach.

        Args:
            image: PIL Image object
            gaps: List of gap regions [(y_start, y_end, gap_size), ...]
            existing_bboxes: Already detected bounding boxes
            img_width: Image width
            img_height: Image height

        Returns:
            List of additional elements found in gaps
        """
        if not gaps:
            return []

        # Convert image to base64
        base64_image = self._encode_image_to_base64(image)

        # Prepare gap information for prompt
        gap_info = "\n".join([f"- Gap at Y coordinates {g[0]}-{g[1]} (size: {g[2]}px)" for g in gaps])

        # Self-critique prompt
        prompt = f"""I previously detected {len(existing_bboxes)} elements in this image.

However, I found suspicious VERTICAL GAPS between elements:
{gap_info}

Please review the image and focus ONLY on these gap regions. Detect any elements (text, labels, signatures, etc.) that are located within these gaps.

IMPORTANT:
- Only return elements within the gap Y-coordinate ranges
- Focus on small elements like labels, field names, or signatures that may have been missed
- Return empty array [] if no elements are in the gaps

Format: [{{"type": "element_type", "bbox": [x1, y1, x2, y2]}}]
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
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            raw_output = response.choices[0].message.content
            return self._parse_json_output(raw_output, img_width, img_height)

        except Exception as e:
            print(f"      [WARNING] Gap detection failed: {e}")
            return []

    def detect_layout_with_refinement(
        self,
        image_input: Union[str, Path, Image.Image],
        gap_threshold: int = 100,
        preprocess: bool = False
    ) -> Dict:
        """
        Detect layout with gap analysis and self-critique refinement.

        This method performs a two-step process:
        1. Initial layout detection
        2. Gap analysis + targeted re-detection in suspicious gaps

        Args:
            image_input: Path to image file or PIL Image object
            gap_threshold: Minimum gap size (pixels) to trigger re-detection
            preprocess: Apply image preprocessing (CLAHE, sharpening) for better detection

        Returns:
            Dictionary containing:
            - raw_output: Raw model response from initial detection
            - elements: Merged list of detected elements (initial + gap-filled)
            - image_dimensions: Original image dimensions
            - refinement_stats: Statistics about the refinement process
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("image_input must be a file path or PIL Image object")

        original_width, original_height = image.size

        print("      [PASS 1/2] Initial layout detection...")
        # Step 1: Initial detection
        initial_result = self.detect_layout(image_input, preprocess=preprocess)
        initial_elements = initial_result['elements']

        print(f"      Initial detection: {len(initial_elements)} elements")

        # Step 2: Analyze gaps
        print("      [PASS 2/2] Analyzing gaps...")
        gaps = self._analyze_gaps(initial_elements, gap_threshold)

        if not gaps:
            print("      No significant gaps found. Skipping refinement.")
            return initial_result

        print(f"      Found {len(gaps)} gaps: {gaps}")

        # Step 3: Detect in gaps
        print(f"      Re-detecting in {len(gaps)} gap regions...")
        existing_bboxes = [elem['bbox'] for elem in initial_elements]
        gap_elements = self._detect_in_gaps(
            image, gaps, existing_bboxes, original_width, original_height
        )

        print(f"      Gap detection found {len(gap_elements)} additional elements")

        # Step 4: Merge results
        merged_elements = self._merge_elements(initial_elements, gap_elements)

        print(f"      Total after refinement: {len(merged_elements)} elements")

        return {
            "raw_output": initial_result['raw_output'],
            "elements": merged_elements,
            "image_dimensions": {"width": original_width, "height": original_height},
            "refinement_stats": {
                "initial_count": len(initial_elements),
                "gaps_found": len(gaps),
                "gap_elements_found": len(gap_elements),
                "final_count": len(merged_elements),
                "elements_added": len(merged_elements) - len(initial_elements)
            }
        }
