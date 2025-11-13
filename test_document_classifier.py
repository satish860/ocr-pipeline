"""Test script for DocumentClassifier - rotation detection and document classification"""

import sys
from pathlib import Path

from src.ocr_pipeline.document_classifier import DocumentClassifier


def test_classifier(image_path: str):
    """
    Test combined rotation detection and document classification.

    Args:
        image_path: Path to test image
    """
    print("=" * 70)
    print("DOCUMENT CLASSIFIER TEST")
    print("=" * 70)
    print(f"\nAnalyzing: {image_path}\n")

    classifier = DocumentClassifier(verbose=True)

    # Combined method - does both rotation detection AND classification
    result = classifier.classify_and_detect_rotation(image_path)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Document Type: {result['document_type'].upper()}")
    print(f"Rotation Needed: {result['rotation_degrees']} degrees")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Reasoning: {result['reasoning']}")
    print()

    # Check expected results for known images
    filename = Path(image_path).name.lower()

    if "cheque" in filename or "check" in filename:
        expected_type = "cheque"
        print(f"Expected Type: {expected_type.upper()}")
        if result['document_type'] == expected_type:
            print("[SUCCESS] Classification correct!")
        else:
            print(f"[FAIL] Expected '{expected_type}' but got '{result['document_type']}'")

    elif "form" in filename or "1040" in filename or "rotated" in filename:
        expected_type = "form"
        print(f"Expected Type: {expected_type.upper()}")
        if result['document_type'] == expected_type:
            print("[SUCCESS] Classification correct!")
        else:
            print(f"[FAIL] Expected '{expected_type}' but got '{result['document_type']}'")

    elif "image" in filename or "chart" in filename:
        expected_type = "general"
        print(f"Expected Type: {expected_type.upper()}")
        if result['document_type'] == expected_type:
            print("[SUCCESS] Classification correct!")
        else:
            print(f"[FAIL] Expected '{expected_type}' but got '{result['document_type']}'")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Test on all sample images in input folder
        print("No image specified. Testing on all samples in input/...\n")

        input_dir = Path("input")
        if input_dir.exists():
            sample_images = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))

            if not sample_images:
                print("Error: No images found in input/ directory")
                sys.exit(1)

            for img_path in sample_images:
                test_classifier(str(img_path))
                print("\n")
        else:
            print("Error: input/ directory not found")
            print("\nUsage: python test_document_classifier.py <image_path>")
            print("\nExample:")
            print("  python test_document_classifier.py input/Cheque.jpg")
            sys.exit(1)
    else:
        test_image = sys.argv[1]

        if not Path(test_image).exists():
            print(f"Error: Image not found: {test_image}")
            sys.exit(1)

        test_classifier(test_image)
