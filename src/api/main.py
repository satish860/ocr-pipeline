"""FastAPI application for OCR Pipeline"""

import io
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from src.ocr_pipeline.image_preprocessor import ImagePreprocessor
from src.ocr_pipeline.layout_detector import LayoutDetector
from src.ocr_pipeline.region_extractor import RegionExtractor
from src.ocr_pipeline.ocr_extractor import OCRExtractor
from src.ocr_pipeline.spatial_analyzer import SpatialAnalyzer

app = FastAPI(
    title="OCR Pipeline API",
    description="Convert images to structured markdown using QwenVL layout detection and Mistral OCR via OpenRouter",
    version="1.0.0"
)

# Initialize pipeline components
preprocessor = ImagePreprocessor()
detector = LayoutDetector()
extractor = RegionExtractor()
ocr = OCRExtractor()
spatial_analyzer = SpatialAnalyzer()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "OCR Pipeline API",
        "version": "1.0.0"
    }


@app.post("/ocr")
async def process_image(
    file: UploadFile = File(...),
    include_annotated_image: bool = True
):
    """
    Process an image through the OCR pipeline.

    Args:
        file: Image file (PNG, JPG, JPEG)
        include_annotated_image: Include base64-encoded annotated image in response

    Returns:
        JSON with markdown content and optionally annotated image
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (PNG, JPG, JPEG)"
        )

    try:
        # Read and load image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Step 1: Preprocess (deskew)
        deskewed_image, angle = preprocessor.deskew_image(image)

        # Step 2: Detect layout with QwenVL (via OpenRouter)
        layout_result = detector.detect_layout(deskewed_image, output_format="json")

        # Step 3: Extract regions
        regions = extractor.extract_regions(deskewed_image, layout_result)

        # Step 4: OCR each region (via OpenRouter Mistral)
        markdown_outputs = []
        for region in regions:
            try:
                markdown_text = ocr.extract_text(
                    region['image'],
                    element_type=region['type']
                )
                markdown_outputs.append({
                    'index': region['index'],
                    'type': region['type'],
                    'bbox': region['bbox'],
                    'markdown': markdown_text
                })
            except Exception as e:
                # Handle individual region failures gracefully
                markdown_outputs.append({
                    'index': region['index'],
                    'type': region['type'],
                    'bbox': region['bbox'],
                    'markdown': f"[Error extracting text: {e}]"
                })

        # Step 5: Analyze spatial relationships
        for i, region in enumerate(regions):
            matching_md = next(
                (md for md in markdown_outputs if md['index'] == region['index']),
                None
            )
            if matching_md:
                region['markdown'] = matching_md['markdown']

        regions_with_relationships = spatial_analyzer.analyze_relationships(regions)
        aligned_groups = spatial_analyzer.group_by_vertical_alignment(
            regions_with_relationships
        )

        # Step 6: Generate combined markdown
        combined_markdown = generate_combined_markdown(
            aligned_groups,
            file.filename
        )

        # Step 7: Optionally create annotated image
        annotated_image_base64 = None
        if include_annotated_image:
            annotated_image_base64 = create_annotated_image(
                deskewed_image,
                layout_result
            )

        # Build response
        response = {
            "success": True,
            "filename": file.filename,
            "detected_elements": len(layout_result['elements']),
            "rotation_correction_degrees": round(angle, 2),
            "markdown": combined_markdown,
            "regions": [
                {
                    "index": md['index'],
                    "type": md['type'],
                    "bbox": md['bbox']
                }
                for md in markdown_outputs
            ]
        }

        if annotated_image_base64:
            response["annotated_image"] = annotated_image_base64

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


def generate_combined_markdown(aligned_groups, filename: str) -> str:
    """Generate spatially-aware combined markdown from aligned groups"""
    lines = []

    processed_indices = set()

    for group in aligned_groups:
        if len(group) > 1:
            types = [r['type'] for r in group]

            if 'table' in types or 'paragraph' in types:
                main_region = group[0]
                related_regions = group[1:]

                lines.append(
                    f"<!-- Region {main_region['index']}: "
                    f"{main_region['type']} {main_region['bbox']} -->\n\n"
                )
                lines.append(main_region.get('markdown', ''))

                if related_regions:
                    lines.append("\n\n**Related content (spatial):**\n")
                    for rel in related_regions:
                        lines.append(f"- *{rel['type']}*: {rel.get('markdown', '')}\n")

                lines.append("\n\n---\n\n")

                for r in group:
                    processed_indices.add(r['index'])
            else:
                for region in group:
                    if region['index'] not in processed_indices:
                        lines.append(
                            f"<!-- Region {region['index']}: "
                            f"{region['type']} {region['bbox']} -->\n\n"
                        )
                        lines.append(region.get('markdown', ''))
                        lines.append("\n\n---\n\n")
                        processed_indices.add(region['index'])
        else:
            region = group[0]
            if region['index'] not in processed_indices:
                lines.append(
                    f"<!-- Region {region['index']}: "
                    f"{region['type']} {region['bbox']} -->\n\n"
                )
                lines.append(region.get('markdown', ''))
                lines.append("\n\n---\n\n")
                processed_indices.add(region['index'])

    return ''.join(lines)


def create_annotated_image(image: Image.Image, layout_result: dict) -> str:
    """Create annotated image with bounding boxes and return as base64"""
    from PIL import ImageDraw, ImageFont

    # Color mapping
    BBOX_COLORS = {
        'table': '#FF0000',
        'paragraph': '#00FF00',
        'header': '#FF00FF',
        'handwritten': '#FF1493',
    }

    # Draw on a copy
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()

    for idx, element in enumerate(layout_result['elements'], 1):
        bbox = element['bbox']
        elem_type = element['type']
        color = BBOX_COLORS.get(elem_type, '#FFFFFF')

        x1, y1, x2, y2 = bbox

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=8)

        # Draw label
        label_bg = [x1, y1 - 50, x1 + 400, y1]
        draw.rectangle(label_bg, fill=color)
        draw.text((x1 + 10, y1 - 45), f"{idx}: {elem_type}", fill='black', font=font)

    # Convert to base64
    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return img_base64


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
