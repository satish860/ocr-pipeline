"""
Convert PDF pages to images.
Usage: uv run python scripts/pdf_to_images.py <path_to_pdf>
"""

import sys
from pathlib import Path
import fitz  # PyMuPDF


def pdf_to_images(pdf_path: str, output_dir: str = None, dpi: int = 300) -> list[Path]:
    """
    Convert each page of a PDF to an image.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save images (default: same as PDF with _images suffix)
        dpi: DPI for image quality (default: 300)

    Returns:
        List of paths to created images
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Create output directory
    if output_dir is None:
        output_dir = pdf_path.parent / f"{pdf_path.stem}_images"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Open PDF
    pdf_document = fitz.open(pdf_path)
    image_paths = []

    print(f"Converting {len(pdf_document)} pages from: {pdf_path.name}")
    print(f"Output directory: {output_dir}")

    # Convert each page
    zoom = dpi / 72  # PyMuPDF uses 72 DPI as base
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # Render page to image
        pix = page.get_pixmap(matrix=matrix)

        # Save as PNG
        image_path = output_dir / f"page_{page_num + 1:04d}.png"
        pix.save(image_path)
        image_paths.append(image_path)

        print(f"  [OK] Page {page_num + 1}/{len(pdf_document)} -> {image_path.name}")

    pdf_document.close()

    print(f"\n[OK] Successfully converted {len(image_paths)} pages")
    print(f"  Images saved to: {output_dir}")

    return image_paths


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/pdf_to_images.py <path_to_pdf> [output_dir] [dpi]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    try:
        pdf_to_images(pdf_path, output_dir, dpi)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
