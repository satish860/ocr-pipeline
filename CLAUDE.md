# OCR Pipeline Project

## Overview
This project implements a **simplified single-stage OCR pipeline** using QwenVL via OpenRouter API:
- **QwenVL (Qwen3-VL-235B)** - Single-pass document extraction with coordinate annotations

**Key Architecture Note**: This is a **lightweight API orchestration** service. QwenVL runs via OpenRouter API, meaning **no local GPU/heavy compute is required**. The application only handles image processing and API coordination.

**Output**: Markdown with inline base64 images + separate image array for special elements (tables, charts, signatures, handwritten text)

## Core Design Principle: OCR vs Extraction

**CRITICAL**: This is an **OCR (Optical Character Recognition) solution**, NOT a data extraction/correction solution.

### What This Means
- **Our job**: Accurately extract what is visible in the image
- **NOT our job**: "Fix", "correct", or "enhance" what we see
- **Goal**: High-fidelity text extraction, not intelligent interpretation

### Validation Philosophy
If validation/correction features are implemented, they MUST follow these rules:

1. **Accuracy First**: Validation must NEVER decrease accuracy
   - If corrections make output worse, validation is broken
   - Always measure: does this improve accuracy?
   - Implement rollback if validation decreases quality

2. **Trust OCR Output**: Default to trusting the OCR extraction
   - OCR might extract MORE complete text than visible (e.g., "IND/LOR VOLUME" vs "VOLUME")
   - More complete text is BETTER, not an error
   - Only flag clear mistakes, not improvements

3. **Quality Gates Required**: Any validation system must include:
   - Measurement of accuracy before/after correction
   - Automatic rollback if corrections decrease accuracy
   - False positive filtering (old_value == new_value)
   - Ground truth comparison when available

### Lessons Learned: Table Validation Failure

**Context**: See `table_bug.md` for full details.

An earlier table validation/correction pipeline decreased accuracy by **87%** (from 89.8% → 2.7% on Sample ID=4) because it:
- Tried to "correct" LaTeX tables by modifying serialized format
- Applied corrections without verifying they improved accuracy
- Trusted AI validator outputs without filtering false positives (26.7% false positive rate)
- Removed valuable information to "match the image exactly"

**Takeaway**: OCR accuracy matters most. Validation that decreases accuracy is worse than no validation.

**Current Status**: Table validation is disabled by default. Any future validation must prove it improves accuracy before being enabled.

## Technology Stack
- Python 3.12
- UV for dependency management
- **QwenVL (Qwen3-VL-235B-A22B-Instruct)** via OpenRouter for document extraction
- Pillow for image processing
- Requests for API calls

## Architecture
```
Input Image → QwenVL API (single call) → Markdown + Extracted Images
```

**Compute Requirements**: CPU-only (lightweight). No GPU needed since all ML inference happens via OpenRouter.

## Development Philosophy
- Test immediately after implementing each feature
- Incremental development approach
- Run code to verify at each step
- Exhaust existing code before writing new code

## API Setup
- Using OpenRouter for QwenVL
- Cost-effective and unified API interface
- Set `OPENROUTER_API_KEY` in `.env` file

## Project Structure
```
ocr-pipeline/
├── src/
│   └── ocr_pipeline/          # Core pipeline modules
│       ├── qwen_extractor.py  # QwenVL extraction logic
│       ├── cli.py             # CLI interface
│       └── __init__.py        # Package exports
├── input/                     # Place input images here (optional)
├── test_qwen_html_parser.py   # Original test script (reference)
└── pyproject.toml             # Dependencies
```

**Note**: No output directory needed - all results are returned in memory.

## Getting Started

### Local Development
1. Copy `.env.example` to `.env` and add your OpenRouter API key
2. Install dependencies: `uv sync`
3. Process an image:
   ```bash
   uv run python -m ocr_pipeline.cli input/document.png
   ```

### Usage (CLI)
```bash
# Process a single image (returns results in memory, does not save files)
python -m ocr_pipeline.cli <image_path>

# Example
python -m ocr_pipeline.cli input/invoice.png
```

### Usage (Python API)
```python
from ocr_pipeline import extract_document

# Extract document
result = extract_document("path/to/image.png", include_images=True)

# Access results
if result['success']:
    # Get markdown with inline images
    markdown = result['markdown']
    print(f"Markdown length: {len(markdown)} characters")

    # Get separate image array
    for img in result['images']:
        print(f"{img['type']}: {img['bbox']}")
        # img['base64'] contains the base64-encoded PNG

    # Get detected elements with coordinates
    for elem in result['elements']:
        print(f"{elem['type']}: {elem['bbox']}")
else:
    print(f"Error: {result['error']}")
```

## Output Format

### Markdown Output
The markdown output includes:
- All text content in natural reading order (top to bottom, left to right)
- HTML alignment tags for right/center-aligned text
- LaTeX tables for structured data
- Coordinate annotations for special elements (0-1000 scale)
- Inline base64 images for charts, tables, signatures, handwritten text

### Extracted Images
Separate array of detected special elements:
```python
[
  {
    "type": "Table",
    "base64": "iVBORw0KGgoAAAANS...",
    "bbox": (100, 200, 800, 600)  # 0-1000 scale
  },
  {
    "type": "Signature",
    "base64": "iVBORw0KGgoAAAANS...",
    "bbox": (50, 850, 300, 950)
  }
]
```

### Return Format
The function returns a dictionary with:
- `success`: Boolean indicating if extraction succeeded
- `markdown`: String with markdown content (includes inline base64 images)
- `images`: Array of extracted images as base64 strings
- `elements`: Array of detected elements with coordinates
- `error`: Error message (if success=False)

**Note**: Nothing is saved to disk. All results are returned in memory.

## Pipeline Flow

```
Input Image
    ↓
┌──────────────────────────────────────────────────────┐
│ Image Preprocessing (enabled by default)            │
│ - Analyze quality (sharpness, contrast)             │
│ - Enhance contrast (+30% if low contrast)           │
│ - Sharpen text edges (unsharp mask)                 │
│ - Upscale if low resolution (respects QwenVL limits)│
│ Overhead: ~100-200ms                                 │
│ Accuracy gain: +4% on low-quality images            │
└──────────────┬───────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────┐
│ QwenVL API Call (single-stage)                      │
│ - Extract all text in reading order                 │
│ - Detect alignment (left/center/right)              │
│ - Convert tables to LaTeX format                    │
│ - Annotate special elements with coordinates        │
│   (tables, images, charts, signatures, handwritten) │
└──────────────┬───────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────┐
│ Post-processing                                      │
│ - Clean markdown wrappers                            │
│ - Add HTML alignment tags (if needed)                │
│ - Extract images from coordinates                    │
│ - Embed images inline as base64                      │
└──────────────┬───────────────────────────────────────┘
               ↓
    Markdown + Extracted Images
```

## Features

### Coordinate Annotations
QwenVL adds HTML comments for special elements:
```markdown
<!-- Table (100, 200, 800, 600) -->
\begin{tabular}{|l|l|l|}
...
\end{tabular}

<!-- Signature (50, 850, 300, 950) -->
John Doe
```

### Alignment Preservation
Text alignment is preserved using HTML tags:
```markdown
<div align="right">
Amount Due: $1,234.56
</div>

<div align="center">
INVOICE
</div>

Regular left-aligned text (no tags needed)
```

### Image Extraction
Special elements are automatically extracted:
- **Tables**: Converted to LaTeX + extracted as PNG
- **Charts/Graphs**: Described in markdown + extracted as PNG
- **Signatures**: Transcribed + extracted as PNG
- **Handwritten text**: Transcribed + extracted as PNG

### Image Preprocessing (Enabled by Default)
All images are automatically preprocessed to improve OCR accuracy:

**Quality Analysis:**
- Sharpness measurement (Laplacian variance)
- Contrast measurement (histogram standard deviation)
- Automatic detection of low-quality images

**Enhancements Applied (if needed):**
- **Contrast Enhancement**: +30% boost for low-contrast images (<50 std dev)
- **Sharpening**: Unsharp mask to enhance text edges
- **Upscaling**: Intelligent upscaling for low-resolution images
  - Targets 70% of QwenVL's max_pixels (1.47M pixels)
  - Respects 2.1M pixel limit
  - Uses Lanczos interpolation (highest quality)

**Performance:**
- Overhead: ~100-200ms per image
- Accuracy improvement: +4% on low-quality images
- No negative impact on high-quality images

**Disable if needed:**
```python
result = extract_document("image.png", preprocess=False)
```

## Environment Variables
- `OPENROUTER_API_KEY`: Required for API access

## Cost Considerations
- QwenVL (235B model) via OpenRouter: ~$0.XX per image
- No compute costs (CPU-only, lightweight)
- No storage costs (results returned in memory only)

## Performance
- **Single image**: ~5-15 seconds (depends on complexity and API latency)
- **Bottleneck**: OpenRouter API latency, not compute
- **No preprocessing overhead**: Single API call replaces 7-stage pipeline

## Advantages Over Previous Multi-Stage Pipeline

### Simplified Architecture
- **Before**: 7 stages (preprocessing → classification → layout detection → region extraction → OCR → spatial analysis → routing)
- **After**: 1 stage (QwenVL single call)
- **Code reduction**: ~2000 lines → ~400 lines (80% reduction)

### Better Accuracy
- **Larger model**: 235B parameters vs 30B (previous layout detector)
- **Native alignment detection**: VLM detects text alignment automatically
- **Better handling of rotated text**: No need for deskewing preprocessing
- **Natural reading order**: VLM understands document structure

### Fewer Dependencies
- **Removed**: opencv-python, pytesseract, numpy, fastapi, uvicorn, pandas, matplotlib, etc.
- **Kept**: pillow, openai, python-dotenv, requests
- **Reduction**: 90% fewer dependencies

## Migration Notes

This codebase was recently migrated from a complex multi-stage pipeline to the simplified QwenVL approach. The previous pipeline included:
- Image preprocessing (deskewing, enhancement)
- Document classification
- Layout detection (Qwen-30B)
- Region extraction
- OCR (Gemini Flash 2.5)
- Spatial analysis
- Smart routing

All of this has been replaced by a single QwenVL API call with an enhanced prompt that handles:
- Text extraction in reading order
- Alignment detection
- Table formatting (LaTeX)
- Coordinate annotations for special elements
- Handwritten text detection

The original test script (`test_qwen_html_parser.py`) is kept for reference.
