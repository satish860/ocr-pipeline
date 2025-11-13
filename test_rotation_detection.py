"""Test script to compare AI vs heuristic rotation detection"""

import sys
from pathlib import Path
from PIL import Image

from src.ocr_pipeline.document_classifier import DocumentClassifier
from src.ocr_pipeline.image_preprocessor import ImagePreprocessor


def test_rotation_detection(image_path: str):
    """
    Test rotation detection on an image using both methods.

    Args:
        image_path: Path to test image
    """
    print("=" * 70)
    print("ROTATION DETECTION TEST")
    print("=" * 70)
    print(f"\nTesting image: {image_path}\n")

    # Load image
    image = Image.open(image_path)
    width, height = image.size
    print(f"Image size: {width}x{height}")
    print(f"Aspect ratio: {width/height:.2f} ({'landscape' if width > height else 'portrait'})\n")

    # Method 1: Current OpenCV Heuristic
    print("-" * 70)
    print("METHOD 1: OpenCV Heuristic (Current)")
    print("-" * 70)

    preprocessor = ImagePreprocessor()
    try:
        heuristic_rotation = preprocessor.detect_orientation(image)
        print(f"[OK] Detected rotation: {heuristic_rotation} degrees")

        if heuristic_rotation == 0:
            print("  -> No rotation needed (document appears upright)")
        else:
            print(f"  -> Needs {heuristic_rotation} degrees counter-clockwise rotation to correct")
    except Exception as e:
        print(f"[ERROR] {e}")
        heuristic_rotation = None

    print()

    # Method 2: AI-Powered Detection (New)
    print("-" * 70)
    print("METHOD 2: AI-Powered Detection (Vision Model) - NEW")
    print("-" * 70)

    classifier = DocumentClassifier()
    try:
        ai_rotation = classifier.detect_rotation(image)
        print(f"[OK] Detected rotation: {ai_rotation} degrees")

        if ai_rotation == 0:
            print("  -> No rotation needed (document appears upright)")
        else:
            print(f"  -> Needs {ai_rotation} degrees counter-clockwise rotation to correct")
    except Exception as e:
        print(f"[ERROR] {e}")
        ai_rotation = None

    print()

    # Comparison
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    if heuristic_rotation is not None and ai_rotation is not None:
        if heuristic_rotation == ai_rotation:
            print(f"[OK] Both methods agree: {ai_rotation} degrees rotation needed")
        else:
            print(f"[WARN] Methods disagree:")
            print(f"  - OpenCV heuristic: {heuristic_rotation} degrees")
            print(f"  - AI detection:     {ai_rotation} degrees")

    print()

    # Expected result for rotated.png
    if "rotated.png" in image_path.lower():
        print("=" * 70)
        print("EXPECTED RESULT FOR rotated.png")
        print("=" * 70)
        print("Expected: 90 degrees (form is rotated 90 degrees clockwise, needs correction)")

        if ai_rotation == 90:
            print("[SUCCESS] AI detection is CORRECT!")
        else:
            print(f"[FAIL] AI detection returned {ai_rotation} degrees (unexpected)")

        if heuristic_rotation == 0:
            print("[FAIL] Heuristic FAILED to detect rotation (returned 0 degrees)")
        elif heuristic_rotation == 90:
            print("[SUCCESS] Heuristic is also correct")
        else:
            print(f"[FAIL] Heuristic returned {heuristic_rotation} degrees (unexpected)")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to testing rotated.png
        test_image = "input/rotated.png"
        print(f"No image specified, using default: {test_image}\n")
    else:
        test_image = sys.argv[1]

    if not Path(test_image).exists():
        print(f"Error: Image not found: {test_image}")
        sys.exit(1)

    test_rotation_detection(test_image)
