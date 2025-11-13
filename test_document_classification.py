"""Test script for document type classification"""

import sys
from pathlib import Path

from src.ocr_pipeline.document_classifier import DocumentClassifier


def test_classification(image_path: str):
    """
    Test document classification on an image.

    Args:
        image_path: Path to test image
    """
    print("=" * 70)
    print("DOCUMENT CLASSIFICATION TEST")
    print("=" * 70)
    print(f"\nTesting image: {image_path}\n")

    classifier = DocumentClassifier()

    # Classify document
    try:
        result = classifier.classify_document_type(image_path)

        print("-" * 70)
        print("CLASSIFICATION RESULT")
        print("-" * 70)
        print(f"Document Type: {result['document_type'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Reasoning: {result['reasoning']}")
        print()

        # Expected results for known images
        if "rotated.png" in image_path.lower() or "1040" in image_path.lower():
            expected = "form"
            print(f"Expected: {expected.upper()}")
            if result['document_type'] == expected:
                print("[SUCCESS] Classification is correct!")
            else:
                print(f"[FAIL] Expected '{expected}' but got '{result['document_type']}'")

        elif "cheque" in image_path.lower() or "check" in image_path.lower():
            expected = "cheque"
            print(f"Expected: {expected.upper()}")
            if result['document_type'] == expected:
                print("[SUCCESS] Classification is correct!")
            else:
                print(f"[FAIL] Expected '{expected}' but got '{result['document_type']}'")

    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Test on all sample images in input folder
        print("No specific image provided. Testing on sample images...\n")

        input_dir = Path("input")
        if input_dir.exists():
            sample_images = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))

            if not sample_images:
                print("No images found in input/ directory")
                sys.exit(1)

            for img_path in sample_images:
                test_classification(str(img_path))
                print("\n")
        else:
            print("Error: input/ directory not found")
            sys.exit(1)
    else:
        test_image = sys.argv[1]

        if not Path(test_image).exists():
            print(f"Error: Image not found: {test_image}")
            sys.exit(1)

        test_classification(test_image)
