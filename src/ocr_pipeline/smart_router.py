"""
Smart Document Router

Adaptive extraction routing based on document type.
Routes to optimal extraction strategy (holistic vs modular) for best accuracy.
"""

from typing import Dict, Any, Union
from pathlib import Path
from PIL import Image

from .document_classifier import DocumentClassifier
from .json_extractor import JSONExtractor
from .layout_detector import LayoutDetector
from .ocr_extractor import OCRExtractor


class SmartDocumentRouter:
    """
    Routes documents to optimal extraction strategy based on type.

    Strategies:
    - Charts/Infographics: Holistic approach (full page image)
    - Tables/Forms: Modular approach (region-based extraction)
    """

    def __init__(self, api_key: str = None, verbose: bool = True):
        """
        Initialize Smart Document Router.

        Args:
            api_key: OpenRouter API key
            verbose: If True, prints routing decisions
        """
        self.verbose = verbose

        # Initialize components
        self.classifier = DocumentClassifier(api_key=api_key, verbose=verbose)
        self.json_extractor = JSONExtractor(api_key=api_key)
        self.layout_detector = LayoutDetector(api_key=api_key)
        self.ocr_extractor = OCRExtractor(api_key=api_key)

    def extract_with_routing(
        self,
        image_input: Union[str, Path, Image.Image],
        json_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract structured JSON using adaptive routing based on document type.

        Args:
            image_input: Path to image file or PIL Image object
            json_schema: JSON schema defining expected structure

        Returns:
            Structured JSON matching the schema
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
            image_path = str(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
            image_path = None
        else:
            raise ValueError("image_input must be a file path or PIL Image object")

        # Step 1: Classify document type
        if self.verbose:
            print("\n[Smart Router] Classifying document type...")

        classification = self.classifier.classify_document_type(image)
        doc_type = classification['document_type']
        confidence = classification['confidence']

        if self.verbose:
            print(f"[Smart Router] Document type: {doc_type.upper()} (confidence: {confidence:.2f})")
            print(f"[Smart Router] Reasoning: {classification['reasoning']}")

        # Step 2: Route to appropriate extraction strategy
        if doc_type in ['chart', 'infographic']:
            # HOLISTIC APPROACH for charts/infographics
            if self.verbose:
                print(f"[Smart Router] -> Using HOLISTIC extraction (full page image)")

            result = self._holistic_extraction(image, image_path, json_schema)

        else:  # table, form, cheque, general
            # MODULAR APPROACH for tables/forms/general
            if self.verbose:
                print(f"[Smart Router] -> Using MODULAR extraction (region-based)")

            result = self._modular_extraction(image, json_schema)

        return result

    def _holistic_extraction(
        self,
        image: Image.Image,
        image_path: str,
        json_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Holistic extraction for charts/infographics.
        Uses full page image without region splitting.

        Args:
            image: PIL Image object
            image_path: Optional path to image file
            json_schema: JSON schema

        Returns:
            Extracted JSON
        """
        # Step 1: Extract chart metadata for better context
        if self.verbose:
            print(f"[Holistic] Extracting chart metadata for context...")

        chart_metadata = self.ocr_extractor.extract_chart_metadata(image)

        # Step 2: Extract basic text from layout detection (don't crop regions)
        # This gives us text overlay but preserves visual context
        layout_result = self.layout_detector.detect_layout(image)

        # Build simple markdown with detected text (no region cropping)
        markdown_parts = []
        for element in layout_result['elements']:
            elem_type = element.get('type', 'unknown')
            # Just note that there's an element here, but don't extract it separately
            markdown_parts.append(f"<!-- Detected {elem_type} element -->")

        markdown = "\n".join(markdown_parts)

        if self.verbose:
            print(f"[Holistic] Detected {len(layout_result['elements'])} elements, sending full page image with metadata")

        # Step 3: Extract JSON with full page image AND chart metadata
        result = self.json_extractor.extract(
            markdown=markdown,
            json_schema=json_schema,
            full_page_image=image,  # ← KEY: Send full page image!
            chart_metadata=chart_metadata  # ← NEW: Send chart metadata for context!
        )

        return result

    def _modular_extraction(
        self,
        image: Image.Image,
        json_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Modular extraction for tables/forms.
        Uses region-based extraction (current approach).

        Args:
            image: PIL Image object
            json_schema: JSON schema

        Returns:
            Extracted JSON
        """
        # Standard pipeline: layout detection → region extraction → OCR → JSON
        # (This is the current approach that works well for tables)

        # For now, this is a simplified version
        # In production, you'd integrate the full pipeline here
        # or call the existing test_layout_detector pipeline

        if self.verbose:
            print(f"[Modular] Using region-based extraction (standard pipeline)")

        # Detect layout
        layout_result = self.layout_detector.detect_layout(image)
        elements = layout_result['elements']

        # Extract regions (simplified - in production use RegionExtractor)
        markdown_parts = []
        for element in elements:
            elem_type = element.get('type', 'text')
            bbox = element.get('bbox', [0, 0, 100, 100])

            # Crop region
            x1, y1, x2, y2 = bbox
            region_image = image.crop((x1, y1, x2, y2))

            # OCR extract
            text = self.ocr_extractor.extract_text(region_image, element_type=elem_type)
            markdown_parts.append(f"<!-- Region: {elem_type} -->\n{text}\n")

        markdown = "\n---\n".join(markdown_parts)

        if self.verbose:
            print(f"[Modular] Extracted {len(elements)} regions")

        # Extract JSON from markdown (no images for modular approach)
        result = self.json_extractor.extract(
            markdown=markdown,
            json_schema=json_schema
        )

        return result


def main():
    """Test the smart router."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ocr_pipeline.smart_router <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Simple test schema
    test_schema = {
        "type": "object",
        "properties": {
            "extracted_data": {"type": "string"}
        }
    }

    router = SmartDocumentRouter(verbose=True)
    result = router.extract_with_routing(image_path, test_schema)

    print("\nExtracted JSON:")
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
