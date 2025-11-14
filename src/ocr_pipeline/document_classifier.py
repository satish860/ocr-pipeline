"""Document classification and rotation detection using Tesseract OSD + VLM fallback"""

import os
import base64
import json
from typing import Dict, Union
from pathlib import Path
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import pytesseract
import platform

# Configure Tesseract path
# Priority: Environment variable > Default Windows path > System PATH
tesseract_path = os.getenv('TESSERACT_PATH')
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
elif platform.system() == 'Windows':
    # Default installation path for Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class DocumentClassifier:
    """
    Detects document rotation and classifies document types.

    Uses hybrid approach:
    - Tesseract OSD for rotation detection (geometric/orientation)
    - VLM for document type classification (semantic understanding)

    Document types: form, cheque, general
    """

    def __init__(self, api_key: str = None, verbose: bool = True):
        """
        Initialize DocumentClassifier with OpenRouter API.

        Args:
            api_key: OpenRouter API key. If None, loads from OPENROUTER_API_KEY env var.
            verbose: If True, prints detection and classification progress. Default: True.
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Provide it via api_key parameter or OPENROUTER_API_KEY env variable."
            )

        self.verbose = verbose

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        # Use Gemini Flash 2.5 for document classification
        # Chosen for: High accuracy on document understanding, good at layout analysis
        self.model = "google/gemini-2.5-flash"

    def classify_and_detect_rotation(self, image_input: Union[str, Path, Image.Image]) -> Dict:
        """
        Combined method: Detect rotation AND classify document type in one call.

        This is the recommended method for efficiency - loads image once and performs both operations.

        Args:
            image_input: Path to image file or PIL Image object

        Returns:
            Dictionary containing:
            - rotation_degrees: 0, 90, 180, or 270
            - document_type: "form", "cheque", or "general"
            - confidence: confidence score (0.0 to 1.0)
            - reasoning: brief explanation of classification

        Example:
            >>> classifier = DocumentClassifier()
            >>> result = classifier.classify_and_detect_rotation('form.png')
            >>> print(f"Type: {result['document_type']}, Rotation: {result['rotation_degrees']}°")
        """
        # Load image once
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("image_input must be a file path or PIL Image object")

        # Detect rotation
        rotation = self.detect_rotation(image)

        # Classify document type
        classification = self.classify_document_type(image)

        # Combine results
        return {
            'rotation_degrees': rotation,
            'document_type': classification['document_type'],
            'confidence': classification['confidence'],
            'reasoning': classification['reasoning']
        }

    def detect_rotation(self, image_input: Union[str, Path, Image.Image]) -> int:
        """
        Detect document rotation angle using hybrid approach.

        Strategy:
        1. Try Tesseract OSD (Orientation and Script Detection) first - fast and reliable
        2. Fall back to VLM with layout-based prompt if OSD fails

        Args:
            image_input: Path to image file or PIL Image object

        Returns:
            Rotation angle in degrees: 0, 90, 180, or 270
            This is the angle needed to CORRECT the orientation
            (i.e., how much to rotate counter-clockwise to make it upright)
        """
        # Load image WITHOUT applying EXIF auto-rotation
        # We want the RAW pixel data so we can detect actual rotation needed
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
            # DO NOT apply EXIF transpose - we want raw orientation
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("image_input must be a file path or PIL Image object")

        # Method 1: Try Tesseract OSD first (most reliable)
        try:
            rotation = self._detect_rotation_tesseract(image)
            if rotation in [0, 90, 180, 270]:
                if self.verbose:
                    print(f"[Rotation Detection] Tesseract OSD detected: {rotation} degrees")
                return rotation
        except Exception as e:
            if self.verbose:
                print(f"[Rotation Detection] Tesseract OSD failed: {e}")
                print("[Rotation Detection] Falling back to VLM detection...")

        # Method 2: Fall back to VLM with improved prompt
        rotation = self._detect_rotation_vlm(image)
        if self.verbose:
            print(f"[Rotation Detection] VLM detected: {rotation} degrees")
        return rotation

    def classify_document_type(self, image_input: Union[str, Path, Image.Image]) -> Dict:
        """
        Classify document type using VLM.

        Args:
            image_input: Path to image file or PIL Image object

        Returns:
            Dictionary with:
            - document_type: "form", "cheque", or "general"
            - confidence: confidence score (0.0 to 1.0)
            - reasoning: brief explanation
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("image_input must be a file path or PIL Image object")

        # Convert image to base64
        base64_image = self._encode_image_to_base64(image)

        # Document classification prompt
        prompt = """Analyze this document image and classify it into ONE of these types:

**1. CHART** - Business charts and graphs:
- Pie charts, donut charts, bar charts, line graphs
- Visual data representation with segments, bars, or data points
- Has chart titles, legends, data labels with values
- Examples: Revenue charts, market share graphs, financial dashboards
- Key indicators: Visual data visualization, chart elements (pie slices, bars, lines)

**2. INFOGRAPHIC** - Complex visual information displays:
- Maps with data overlays, flowcharts, process diagrams
- Multiple visual elements combined (maps + charts + icons)
- Designed for visual storytelling or data presentation
- Examples: Geographic revenue maps, organizational charts, process flows
- Key indicators: Multiple visual elements, decorative graphics, spatial layout matters

**3. TABLE** - Structured tabular data:
- Rows and columns with clear grid structure
- Data organized in cells with headers
- Minimal graphics, primarily text/numbers in table format
- Examples: Financial tables, data grids, comparison tables, spreadsheets
- Key indicators: Clear row/column structure, cell borders, header row

**4. FORM** - Documents with form structure:
- Has form field labels (e.g., "Name:", "Date:", "Address:", "SSN:")
- Has input fields, checkboxes, or blank spaces for filling
- Examples: Tax forms (1040, W-2), application forms, surveys, registration forms
- Key indicators: Structured layout with labels and fill-in areas

**5. CHEQUE** - Bank cheques/checks:
- Has cheque-specific fields: "Pay to the order of", amount fields (numeric and written)
- Has MICR line at bottom (magnetic ink characters)
- Has bank name, cheque number, date, signature line
- Examples: Personal cheques, business cheques
- Key indicators: Standard cheque format with payee, amount, bank details

**6. GENERAL** - All other documents:
- Invoices, receipts, letters, reports, articles, flyers
- Standard documents without specialized visual structure
- Examples: Invoices, business letters, reports, receipts, product catalogs
- Key indicators: Flowing text or simple layout without specialized structure

IMPORTANT:
- Prioritize CHART if you see pie/donut/bar charts even if there are tables nearby
- Classify as INFOGRAPHIC if multiple visual elements (map + chart, diagram + icons)
- Classify as TABLE only if the ENTIRE document is primarily a data grid
- Classify as FORM only if it has BOTH labels AND fillable fields
- When in doubt between types, choose the most visually dominant element

Return ONLY this JSON:
{
  "document_type": "chart" or "infographic" or "table" or "form" or "cheque" or "general",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation of why you chose this type"
}"""

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
                ],
                temperature=0.0  # Deterministic for classification
            )

            raw_output = response.choices[0].message.content

            # Parse response
            classification = self._parse_classification_output(raw_output)

            if self.verbose:
                print(f"[Document Classification] Type: {classification['document_type']}, "
                      f"Confidence: {classification['confidence']:.2f}")
                print(f"[Classification Reasoning] {classification['reasoning']}")

            return classification

        except Exception as e:
            raise RuntimeError(f"Error calling OpenRouter API for classification: {e}")

    def _detect_rotation_tesseract(self, image: Image.Image) -> int:
        """
        Use Tesseract OSD (Orientation and Script Detection) to detect rotation.

        Args:
            image: PIL Image object

        Returns:
            Rotation angle: 0, 90, 180, or 270
        """
        try:
            # Run Tesseract OSD
            osd_data = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)

            # Get rotation angle
            # Tesseract returns the detected rotation (how much the image IS rotated)
            # We need to return the correction angle
            detected_rotation = osd_data.get('rotate', 0)

            # Convert to correction angle (counter-clockwise rotation needed)
            # If image is rotated 90° clockwise, we need 270° counter-clockwise (or 90° clockwise correction)
            if detected_rotation == 0:
                return 0
            elif detected_rotation == 90:
                return 270  # Image rotated 90° CW → need 270° CCW correction
            elif detected_rotation == 180:
                return 180
            elif detected_rotation == 270:
                return 90  # Image rotated 270° CW → need 90° CCW correction
            else:
                return 0

        except Exception as e:
            raise RuntimeError(f"Tesseract OSD failed: {e}")

    def _detect_rotation_vlm(self, image: Image.Image) -> int:
        """
        Use VLM with layout-based prompt as fallback.

        Args:
            image: PIL Image object

        Returns:
            Rotation angle: 0, 90, 180, or 270
        """
        # Convert image to base64
        base64_image = self._encode_image_to_base64(image)

        # Improved layout-based prompt (better than asking about text readability)
        prompt = """You are analyzing a PHYSICAL document image. Look at VISUAL LAYOUT, not content understanding.

CRITICAL: VLMs can read upside-down text, so don't rely on readability. Focus on GEOMETRIC LAYOUT.

Analyze these VISUAL markers:
1. Where is the main title/header located? (top, bottom, left edge, right edge)
2. For forms: Where are signature lines typically? (bottom usually)
3. Page numbers: Where are they? (usually bottom)
4. For tax forms like 1040: Title should be at TOP, signature at BOTTOM when correct

Based on standard document conventions:
- Title at TOP, content flows downward = 0 degrees (correct)
- Title at BOTTOM, everything upside down = 180 degrees (needs flip)
- Content sideways, read by tilting head LEFT = 90 degrees
- Content sideways, read by tilting head RIGHT = 270 degrees

Think step-by-step about VISUAL LAYOUT STRUCTURE, not text semantics.

For Form 1040 specifically: "U.S. Individual Income Tax Return" header should be at the TOP.

Return ONLY this JSON:
{"rotation_degrees": 0 or 90 or 180 or 270, "reasoning": "brief explanation of layout observation"}"""

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
                ],
                temperature=0.0  # Deterministic output for classification
            )

            raw_output = response.choices[0].message.content

            # Parse JSON response
            rotation_data = self._parse_rotation_output(raw_output)

            return rotation_data['rotation_degrees']

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

    def _parse_rotation_output(self, raw_output: str) -> Dict:
        """
        Parse rotation detection output from VLM.

        Args:
            raw_output: Raw model response

        Returns:
            Dictionary with rotation_degrees and reasoning keys
        """
        # Extract JSON from markdown code blocks if present
        if '```json' in raw_output:
            raw_output = raw_output.split('```json')[1].split('```')[0].strip()
        elif '```' in raw_output:
            raw_output = raw_output.split('```')[1].split('```')[0].strip()

        try:
            data = json.loads(raw_output)

            # Validate rotation value
            rotation = data.get('rotation_degrees', 0)
            if rotation not in [0, 90, 180, 270]:
                raise ValueError(f"Invalid rotation value: {rotation}. Must be 0, 90, 180, or 270")

            reasoning = data.get('reasoning', 'No reasoning provided')
            if self.verbose:
                print(f"[VLM Reasoning] {reasoning}")

            return {'rotation_degrees': rotation, 'reasoning': reasoning}

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse rotation output: {e}\nOutput: {raw_output[:500]}")

    def _parse_classification_output(self, raw_output: str) -> Dict:
        """
        Parse document classification output from VLM.

        Args:
            raw_output: Raw model response

        Returns:
            Dictionary with document_type, confidence, and reasoning keys
        """
        # Extract JSON from markdown code blocks if present
        if '```json' in raw_output:
            raw_output = raw_output.split('```json')[1].split('```')[0].strip()
        elif '```' in raw_output:
            raw_output = raw_output.split('```')[1].split('```')[0].strip()

        try:
            data = json.loads(raw_output)

            # Validate document type
            doc_type = data.get('document_type', 'general').lower()  # Convert to lowercase
            valid_types = ['chart', 'infographic', 'table', 'form', 'cheque', 'general']
            if doc_type not in valid_types:
                if self.verbose:
                    print(f"[WARNING] Invalid document type '{doc_type}', defaulting to 'general'")
                doc_type = 'general'

            # Validate confidence
            confidence = data.get('confidence', 0.5)
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                confidence = 0.5

            reasoning = data.get('reasoning', 'No reasoning provided')

            return {
                'document_type': doc_type,
                'confidence': float(confidence),
                'reasoning': reasoning
            }

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse classification output: {e}\nOutput: {raw_output[:500]}")
