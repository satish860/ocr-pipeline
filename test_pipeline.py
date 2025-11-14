"""Test full pipeline: Classification -> Preprocessing -> Layout Detection"""

from PIL import Image, ImageDraw, ImageFont
from src.ocr_pipeline.document_classifier import DocumentClassifier
from src.ocr_pipeline.preprocessing_pipeline import PreprocessingPipeline
from src.ocr_pipeline.layout_detector import LayoutDetector
import os

# Load specific image
image_path = "input/cnn_0003.png"
print(f"Testing with: {image_path}\n")

image = Image.open(image_path)
print(f"Original image size: {image.size}")

# Stage 1: Document Classification
print("\n=== Stage 1: Document Classification ===")
classifier = DocumentClassifier()
classification = classifier.classify_and_detect_rotation(image)

print(f"Document type: {classification['document_type']}")
print(f"Rotation needed: {classification['rotation_degrees']}°")
print(f"Confidence: {classification['confidence']}")

# Stage 2: Preprocessing
print("\n=== Stage 2: Preprocessing ===")
preprocessor = PreprocessingPipeline()
preprocessed_image, metadata = preprocessor.preprocess(
    image, 
    rotation_degrees=classification['rotation_degrees']
)

print(f"Rotation applied: {metadata['rotation_applied']}°")
print(f"Deskew angle: {metadata['deskew_angle']:.2f}°")
print(f"Preprocessed size: {preprocessed_image.size}")

# Save preprocessed image
os.makedirs("output/test_pipeline", exist_ok=True)
preprocessed_path = "output/test_pipeline/preprocessed_image.png"
preprocessed_image.save(preprocessed_path)
print(f"Saved: {preprocessed_path}")

# Stage 3: Layout Detection
print("\n=== Stage 3: Layout Detection ===")
detector = LayoutDetector()
layout = detector.detect_layout(preprocessed_image)

print(f"Detected elements: {len(layout['elements'])}")
print(f"\nDetected layout elements:")
for i, elem in enumerate(layout['elements'], 1):
    print(f"  {i:2d}. {elem['type']:20s} | BBox: {elem['bbox']}")

# Stage 4: Create annotated image with bounding boxes
print("\n=== Stage 4: Creating Annotated Image ===")
annotated_image = preprocessed_image.copy()
draw = ImageDraw.Draw(annotated_image)

# Define colors for different element types
colors = {
    'segment': 'red',  # Default for simplified segment detection
    'header': 'blue',
    'paragraph': 'green',
    'table': 'cyan',
    'signature': 'purple',
    'signature_box': 'orange',
    'label': 'yellow',
    'handwritten': 'magenta',
    'box': 'pink'
}

for i, elem in enumerate(layout['elements'], 1):
    bbox = elem['bbox']
    elem_type = elem['type']
    color = colors.get(elem_type, 'red')  # Default to red if type not found

    # Draw rectangle
    draw.rectangle(bbox, outline=color, width=3)

    # Draw label
    label = f"{i}: {elem_type}"
    draw.text((bbox[0], bbox[1] - 20), label, fill=color)

# Save annotated image
annotated_path = "output/test_pipeline/layout_annotated.png"
annotated_image.save(annotated_path)
print(f"Saved: {annotated_path}")

print(f"\n[OK] Pipeline test completed!")
print(f"[OK] Check output/test_pipeline/ for:")
print(f"    - preprocessed_image.png (preprocessed)")
print(f"    - layout_annotated.png (with bounding boxes)")
