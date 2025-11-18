"""
FastAPI server for OCR Pipeline.

Provides REST API endpoints for processing images and PDFs with OCR.
"""

import io
import base64
from pathlib import Path
from typing import Optional, Union, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from ocr_pipeline import OCRPipeline
from ocr_pipeline.pdf_extractor import process_pdf, _convert_to_serializable

# Initialize FastAPI app
app = FastAPI(
    title="OCR Pipeline API",
    description="REST API for document OCR processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR Pipeline (with default settings)
ocr_pipeline = OCRPipeline(
    preprocess=True,
    refine=False,
    agentic_refine=False,
    auto_refine=False,
    convert_tables_to_html=False
)

class ImageProcessingResponse(BaseModel):
    """Response model for image processing results."""
    success: bool
    markdown: str
    images: list = []
    elements: list = []
    usage: dict = {}
    quality: Optional[dict] = None
    refinement: Optional[dict] = None
    refinement_analysis: Optional[dict] = None
    error: Optional[str] = None

class PDFProcessingResponse(BaseModel):
    """Response model for PDF processing results."""
    success: bool
    markdown: str
    images: list = []
    elements: list = []
    usage: dict = {}
    quality: Optional[dict] = None
    refinement: Optional[dict] = None
    refinement_analysis: Optional[dict] = None
    error: Optional[str] = None
    page_number: Optional[int] = None

@app.post("/process-image", response_model=ImageProcessingResponse)
async def process_image(
    file: UploadFile = File(...),
    preprocess: bool = Query(True),
    refine: bool = Query(False),
    agentic_refine: bool = Query(False),
    auto_refine: bool = Query(False),
    convert_tables_to_html: bool = Query(False),
    include_images: bool = Query(True),
    include_usage: bool = Query(False),
):
    """
    Process an image file with OCR.
    
    Args:
        file: Image file to process (PNG, JPG, etc.)
        preprocess: Whether to apply image preprocessing
        refine: Whether to refine extraction using Claude
        agentic_refine: Whether to use iterative agentic refinement
        auto_refine: Whether to automatically refine if needed
        convert_tables_to_html: Whether to convert LaTeX tables to HTML
        include_images: Whether to extract and embed images
        include_usage: Whether to include usage/cost data
    
    Returns:
        Processing result with markdown, images, and metadata
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (PNG, JPG, etc.)"
            )
        
        # Read file content
        content = await file.read()
        
        # Open image
        image = Image.open(io.BytesIO(content))
        
        # Create pipeline with specified options
        pipeline = OCRPipeline(
            preprocess=preprocess,
            refine=refine,
            agentic_refine=agentic_refine,
            auto_refine=auto_refine,
            convert_tables_to_html=convert_tables_to_html
        )
        
        # Process image
        result = pipeline.apply(
            image,
            include_images=include_images,
            include_usage=include_usage
        )
        
        # Convert to serializable format (handles numpy types)
        return _convert_to_serializable(result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/process-pdf", response_model=List[PDFProcessingResponse])
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    preprocess: bool = Query(True),
    refine: bool = Query(False),
    agentic_refine: bool = Query(False),
    auto_refine: bool = Query(False),
    convert_tables_to_html: bool = Query(True),
    dpi: int = Query(300, ge=72, le=600),
    output_folder: Optional[Union[str, Path]] = None,
    save_images: bool = Query(False),
    max_refinement_iterations: int = Query(2),
    max_workers: int = Query(4),
    save_results: bool = Query(False),
):
    """
    Process a PDF file with OCR.
    
    Converts PDF pages to images and processes each with OCR.
    
    Args:
        file: PDF file to process
        preprocess: Whether to apply image preprocessing
        refine: Whether to refine extraction using Claude
        agentic_refine: Whether to use iterative agentic refinement
        auto_refine: Whether to automatically refine if needed
        convert_tables_to_html: Whether to convert LaTeX tables to HTML
        include_images: Whether to extract and embed images
        include_usage: Whether to include usage/cost data
        dpi: DPI for PDF to image conversion (72-600)
        max_pages: Maximum number of pages to process (None = all)
    
    Returns:
        List of processing results, one per page
    """
    try:
        # Validate file type
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400,
                detail="File must be a PDF"
            )
        
        # Read file content
        content = await file.read()
        
        # Save PDF temporarily
        temp_pdf_path = Path("/tmp") / f"temp_{file.filename}"
        temp_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_pdf_path, "wb") as f:
            f.write(content)
        
        try:
            # Use process_pdf to handle PDF conversion and OCR processing
            results = process_pdf(
                pdf_path=temp_pdf_path,
                output_folder=None,
                save_images=False,
                preprocess=preprocess,
                refine=refine,
                agentic_refine=agentic_refine,
                auto_refine=auto_refine,
                convert_tables_to_html=convert_tables_to_html,
                max_workers=max_workers,
                save_results=save_results,
                dpi=dpi,
            )
            
            return results
            
        finally:
            # Clean up temporary file
            if temp_pdf_path.exists():
                temp_pdf_path.unlink()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
