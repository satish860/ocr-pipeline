"""
PDF to Text Extraction Module with Enhanced Progress Tracking.
This module provides functionality to extract text from PDF files by converting them
to images and processing them with QwenVL OCR.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from tqdm import tqdm

from ocr_pipeline import OCRPipeline
from .retry_utils import retry_with_backoff

@retry_with_backoff(max_retries=3, initial_delay=0.5, backoff_factor=2.0)
def _process_single_page(args: Tuple) -> Tuple[int, Image.Image]:
    """
    Worker function to process a single PDF page with retry logic.
    This runs in a separate thread.
    
    Args:
        args: Tuple of (pdf_document, page_num, dpi)
    
    Returns:
        Tuple of (page_num, PIL Image)
    """
    pdf_document, page_num, dpi = args
    
    # Load and convert the page
    page = pdf_document.load_page(page_num)
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    return page_num, img


def convert_pdf_to_images(
    pdf_path: Union[str, Path],
    output_folder: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    fmt: str = 'png',
    max_workers: Optional[int] = None
) -> List[Image.Image]:
    """
    Convert a PDF file to a list of PIL Images using parallel processing with threads.
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Optional folder to save the images. If None, images won't be saved.
        dpi: DPI for the output images
        fmt: Output image format (png, jpeg, etc.)
        max_workers: Maximum number of worker threads. If None, uses CPU count * 2.
    
    Returns:
        List of PIL Image objects (in page order)
    """
    # Convert to Path object if it's a string
    pdf_path = Path(pdf_path) if isinstance(pdf_path, str) else pdf_path
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Create output folder if needed
    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
    # Open the PDF once and share it across threads
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    
    try:
        # Prepare arguments for each page
        page_args = [(pdf_document, page_num, dpi) for page_num in range(total_pages)]
        
        # Default to CPU count * 2 for I/O-bound tasks
        if max_workers is None:
            max_workers = (os.cpu_count() or 1) * 2
        
        print(f"📄 Converting PDF to images ({total_pages} pages) using {max_workers} threads...")
        
        # Process pages in parallel using threads
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(_process_single_page, args): args[1] 
                       for args in page_args}
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), 
                              desc="Converting pages", unit="page"):
                page_num, img = future.result()
                results[page_num] = img
                
                # Save image if output folder is provided
                if output_folder is not None:
                    img_path = output_folder / f"page_{page_num + 1:03d}.{fmt}"
                    img.save(img_path, format=fmt.upper(), dpi=(dpi, dpi))
        
        # Sort results by page number to maintain order
        images = [results[i] for i in range(total_pages)]
        
        print(f"✓ Converted {len(images)} pages to images\n")
        return images
        
    finally:
        # Make sure to close the document
        pdf_document.close()


@retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
def _process_page(page_data: tuple) -> Dict[str, Any]:
    """
    Process a single page image using OCR Pipeline with retry logic.
    
    Args:
        page_data: Tuple of (page_number, image, pipeline_config)
    
    Returns:
        Extraction result for the page
    """
    page_number, img, pipeline_config = page_data
    start_time = time.time()
    
    # Initialize OCR pipeline with configuration
    pipeline = OCRPipeline(**pipeline_config)

    # Apply OCR pipeline directly with the PIL Image object
    result = pipeline.apply(
        image_path=img,
        include_images=True,
        include_usage=False
    )
    
    # Add page number to the result
    result['page_number'] = page_number
    
    print(f"Page {page_number} processed in {round(time.time() - start_time, 2)}s")
    return result


def _convert_to_serializable(obj: Any) -> Any:
    """
    Convert NumPy and other non-serializable types to JSON-serializable types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_results_to_file(results: List[Dict[str, Any]], output_folder: Optional[Union[str, Path]] = None):
    """
    Save the results to a file.
    
    Args:
        results: List of extraction results for each page
        output_folder: Optional folder to save the results
    """
    if output_folder is None:
        output_folder = Path.cwd()
    else:
        # Convert to Path object if it's a string
        output_folder = Path(output_folder) if isinstance(output_folder, str) else output_folder
    
    # Create output folder if needed
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Convert results to JSON-serializable format
    serializable_results = _convert_to_serializable(results)
    
    # Save results to a file
    output_path = output_folder / 'results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_path}")


def process_pdf(
    pdf_path: Union[str, Path],
    output_folder: Optional[Union[str, Path]] = "./output",
    save_images: bool = False,
    preprocess: bool = True,
    refine: bool = False,
    agentic_refine: bool = False,
    auto_refine: bool = False,
    max_refinement_iterations: int = 2,
    convert_tables_to_html: bool = True,
    max_workers: int = 4,
    **extract_kwargs
) -> List[Dict[str, Any]]:
    """
    Process a PDF file and extract text using OCR Pipeline with parallel processing.
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Optional folder to save the extracted images and results
        save_images: Whether to save the extracted page images
        preprocess: Whether to apply image preprocessing (default: True)
        refine: Whether to refine extraction using Claude (default: False)
        agentic_refine: Whether to use iterative agentic refinement (default: False)
        auto_refine: Whether to automatically refine only if content needs it (default: False)
        max_refinement_iterations: Maximum iterations for agentic refinement (default: 2)
        convert_tables_to_html: Whether to convert LaTeX tables to HTML (default: True)
        max_workers: Number of worker threads for parallel processing (default: 4)
        **extract_kwargs: Additional arguments (e.g., dpi, min_pixels, max_pixels)
    
    Returns:
        List of extraction results for each page with OCR pipeline processing
    """
    overall_start = time.time()

    # Convert PDF to images
    images = convert_pdf_to_images(
        pdf_path=pdf_path,
        output_folder=output_folder if save_images else None,
        dpi=extract_kwargs.pop('dpi', 300)
    )
    
    # Build OCR pipeline configuration
    pipeline_config = {
        'preprocess': preprocess,
        'refine': refine,
        'agentic_refine': agentic_refine,
        'auto_refine': auto_refine,
        'max_refinement_iterations': extract_kwargs.pop('max_refinement_iterations', 2),
        'convert_tables_to_html': extract_kwargs.pop('convert_tables_html', True),
        'output_dir': output_folder,
        'min_pixels': extract_kwargs.pop('min_pixels', 512 * 32 * 32),
        'max_pixels': extract_kwargs.pop('max_pixels', 4608 * 32 * 32),
    }
    
    # Prepare page data for parallel processing
    page_data = [
        (i, img, pipeline_config)
        for i, img in enumerate(images, 1)
    ]
    
    print(f"🔍 Processing {len(images)} pages with OCR Pipeline...")
    
    # Process pages in parallel using ThreadPoolExecutor with progress bar
    results = [None] * len(images)  # Pre-allocate list to maintain order
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(_process_page, data): idx 
            for idx, data in enumerate(page_data)
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(images), desc="Processing pages", unit="page") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    
                    # Update progress bar with status
                    page_num = result['page_number']
                    status = "✓" if result.get('success', True) else "✗"
                    time_taken = result.get('processing_time', 0)
                    pbar.set_postfix_str(f"Page {page_num} {status} ({time_taken:.1f}s)")
                    pbar.update(1)
                    
                except Exception as e:
                    # Handle any exceptions during processing
                    page_num = page_data[idx][0]
                    results[idx] = {
                        'page_number': page_num,
                        'success': False,
                        'error': str(e),
                        'processing_time': 0
                    }
                    pbar.set_postfix_str(f"Page {page_num} ✗ ERROR")
                    pbar.update(1)
                    print(f"\n⚠️  Error processing page {page_num}: {e}")

    # Save results to a file
    print("\n💾 Saving results...")
    save_results_to_file(results, output_folder)
    
    # Print summary
    total_time = time.time() - overall_start
    print(f"\n✓ PDF processing complete. {len(results)} pages processed in {total_time:.2f}s.")    
    return results