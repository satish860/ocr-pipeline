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

### Refinement Philosophy
The Claude refiner aggressively improves OCR extraction quality by fixing ALL errors:

1. **Aggressive Refinement**: Fix ALL errors where extraction doesn't match the image
   - Don't be conservative - if structure is wrong (especially tables), rebuild it completely
   - Table structures are frequently wrong and need complete reconstruction
   - Verify by visually comparing against the original image

2. **Preserve Valuable Information**: Keep text that is MORE complete than visible
   - If OCR extracted "IND/LOR VOLUME" vs just "VOLUME", that's BETTER - keep it
   - Don't "correct" the document content itself (if invoice says "9,04,116" don't change to "904116")
   - Expanded abbreviations are good (e.g., "Number" vs "No.")

3. **Quality Gates Required**: Refinement system includes:
   - Visual validation against original image
   - Structure validation (table column counts, rowspan/colspan correctness)
   - Automatic rollback if refinement looks broken
   - Measurement of improvement when ground truth available

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
- **Claude Sonnet 4.5** via Anthropic API for optional refinement
- Pillow for image processing and preprocessing
- OpenAI Python library for OpenRouter API calls
- Requests for HTTP API calls
- Datasets library for evaluation

## Architecture
```
Input Image → [Image Preprocessing] → QwenVL API → [Table Conversion] → [Claude Refinement] → Markdown + Extracted Images
```

**Compute Requirements**: CPU-only (lightweight). No GPU needed since all ML inference happens via APIs.

## Development Philosophy
- Test immediately after implementing each feature
- Incremental development approach
- Run code to verify at each step
- Exhaust existing code before writing new code

## API Setup
- Using OpenRouter for QwenVL: Set `OPENROUTER_API_KEY` in `.env` file
- Using Anthropic API for Claude refinement: Set `ANTHROPIC_API_KEY` in `.env` file (optional)
- Cost-effective API-based approach with no local GPU requirements

## Project Structure
```
ocr-pipeline/
├── src/
│   └── ocr_pipeline/          # Core pipeline modules
│       ├── __init__.py        # Package exports (extract_document function)
│       ├── cli.py             # CLI interface
│       ├── ocr_pipeline.py    # Main pipeline orchestrator
│       ├── image_analyzer.py  # Image quality analysis and preprocessing
│       ├── qwen_extractor.py  # QwenVL API integration
│       ├── table_converter.py # LaTeX to HTML table conversion
│       └── claude_refiner.py  # Claude Sonnet 4.5 refinement (optional)
├── input/                     # Place input images here (optional)
├── output/                    # Output directory (only if output_dir specified)
├── test_qwen_html_parser.py   # Original test script (reference)
└── pyproject.toml             # Dependencies
```

**Note**: Results are returned in memory by default. Files are only saved if `output_dir` is specified.

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
# Basic extraction
python -m ocr_pipeline.cli <image_path>

# With preprocessing disabled
python -m ocr_pipeline.cli <image_path> --no-preprocess

# With Claude refinement (single-pass)
python -m ocr_pipeline.cli <image_path> --refine

# With agentic refinement loop (iterative with convergence detection)
python -m ocr_pipeline.cli <image_path> --agentic-refine --max-iterations 3

# With table conversion to HTML
python -m ocr_pipeline.cli <image_path> --convert-tables

# Example
python -m ocr_pipeline.cli input/invoice.png --refine
```

### Usage (Python API)
```python
from ocr_pipeline import extract_document

# Basic extraction
result = extract_document("path/to/image.png", include_images=True)

# With preprocessing disabled
result = extract_document("path/to/image.png", preprocess=False)

# With Claude refinement (single-pass)
result = extract_document(
    "path/to/image.png",
    refine=True,
    include_usage=True  # Include token usage/cost data
)

# With agentic refinement loop (iterative with convergence detection)
result = extract_document(
    "path/to/image.png",
    agentic_refine=True,
    max_refinement_iterations=3,
    include_usage=True
)

# With table conversion to HTML
result = extract_document(
    "path/to/image.png",
    convert_tables_to_html=True
)

# Save intermediate outputs (QwenVL + each refinement iteration)
result = extract_document(
    "path/to/image.png",
    agentic_refine=True,
    output_dir="output"  # Saves timestamped files for each stage
)

# Access results
if result['success']:
    # Get markdown (LaTeX tables or HTML if refined/converted)
    markdown = result['markdown']
    print(f"Markdown length: {len(markdown)} characters")

    # Get separate image array
    for img in result['images']:
        print(f"{img['type']}: {img['bbox']}")
        # img['base64'] contains base64-encoded PNG

    # Get detected elements with coordinates
    for elem in result['elements']:
        print(f"{elem['type']}: {elem['bbox']}")

    # Check quality metrics (if preprocess=True)
    if result.get('quality'):
        print(f"Sharpness: {result['quality']['sharpness']}")
        print(f"Contrast: {result['quality']['contrast']}")
        print(f"Preprocessing applied: {result['quality']['preprocessing_applied']}")

    # Check refinement info (if refine=True or agentic_refine=True)
    if result.get('refinement'):
        print(f"Iterations: {result['refinement']['iterations']}")
        print(f"Converged: {result['refinement']['converged']}")
        if 'final_score' in result['refinement']:
            print(f"Final score: {result['refinement']['final_score']}")

    # Check usage/cost data (if include_usage=True)
    if result.get('usage'):
        print(f"Total cost: ${result['usage'].get('total_cost', 0):.4f}")
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
  - LaTeX tables by default
  - HTML tables if `convert_tables_to_html=True` or `refine=True`
- `images`: Array of extracted images as base64 strings
- `elements`: Array of detected elements with coordinates
- `quality`: Image quality metrics (if `preprocess=True`)
  - `sharpness`: Laplacian variance score
  - `contrast`: Histogram standard deviation
  - `width`, `height`: Image dimensions
  - `needs_preprocessing`: Boolean
  - `preprocessing_applied`: Boolean
- `refinement`: Refinement metadata (if `refine=True` or `agentic_refine=True`)
  - `iterations`: Number of refinement iterations performed
  - `converged`: Whether convergence was achieved
  - `final_score`: Similarity score (if available)
  - `iteration_history`: List of iteration details
- `usage`: Token usage and cost data (if `include_usage=True`)
  - `qwen_tokens`: QwenVL token count
  - `claude_tokens`: Claude token count (if refinement used)
  - `total_cost`: Estimated total cost in USD
- `error`: Error message (if success=False)

**Note**: Results are returned in memory. Files are only saved if `output_dir` is specified in the pipeline constructor.

## Pipeline Flow

```
Input Image
    ↓
┌──────────────────────────────────────────────────────┐
│ Image Preprocessing (optional, enabled by default)  │
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
┌──────────────────────────────────────────────────────┐
│ Table Conversion (optional)                          │
│ - Convert LaTeX tables to HTML format                │
│ - Preserve table structure and content               │
└──────────────┬───────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────┐
│ Claude Refinement (optional)                         │
│                                                       │
│ Single-pass mode (refine=True):                      │
│ - One refinement iteration with Claude Sonnet 4.5    │
│ - Fixes OCR errors, improves table structure         │
│ - ~10-30 seconds                                      │
│                                                       │
│ Agentic loop mode (agentic_refine=True):             │
│ - Iterative refinement with convergence detection    │
│ - Visual validation against original image           │
│ - Structure validation (table column counts, etc.)   │
│ - Automatic rollback if refinement degrades quality  │
│ - Smart stopping when convergence achieved           │
│ - 2-3 iterations typical, ~30-90 seconds             │
└──────────────┬───────────────────────────────────────┘
               ↓
    Markdown + Extracted Images
```

## Features

### Claude Refinement (Optional)

Two refinement modes are available to improve OCR accuracy:

**Single-Pass Refinement (`refine=True`)**:
- One refinement iteration with Claude Sonnet 4.5
- Fixes OCR errors, improves table structure
- Converts LaTeX tables to HTML automatically
- ~10-30 seconds additional processing time
- Good for general accuracy improvement

**Agentic Refinement Loop (`agentic_refine=True`)**:
- Iterative refinement with smart stopping
- **Phase 1**: Basic refinement loop
  - Multiple iterations with Claude Sonnet 4.5
  - Configurable max iterations (default: 2)
  - Saves intermediate outputs if `output_dir` specified
- **Phase 2**: Smart convergence detection (implemented)
  - Measures similarity between iterations using difflib
  - Automatic stop when changes < 5% (converged)
  - Prevents unnecessary iterations and cost
  - Iteration history tracking
- **Future Phase 3**: Quality gates and rollback
  - Visual validation against original image
  - Structure validation (table column counts, etc.)
  - Automatic rollback if refinement degrades quality
  - Accuracy measurement when ground truth available
- Best for critical documents requiring highest accuracy
- ~30-90 seconds typical (2-3 iterations)

**Usage Example**:
```python
# Single-pass refinement
result = extract_document("invoice.png", refine=True)

# Agentic refinement with convergence detection
result = extract_document(
    "invoice.png",
    agentic_refine=True,
    max_refinement_iterations=3,
    output_dir="output"  # Save each iteration
)

# Check refinement results
if result.get('refinement'):
    print(f"Converged: {result['refinement']['converged']}")
    print(f"Iterations: {result['refinement']['iterations']}")
```

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
- `OPENROUTER_API_KEY`: Required for QwenVL API access
- `ANTHROPIC_API_KEY`: Optional, required only for Claude refinement

## Cost Considerations
- **QwenVL**: ~$0.02-0.05 per image (varies by resolution)
- **Claude refinement**: ~$0.03-0.10 per iteration (varies by content length)
- **Total typical cost**: $0.02-0.15 per document (depending on options)
- **Compute**: Minimal (CPU-only, no GPU needed)
- **Storage**: Results in memory (no disk usage unless `output_dir` specified)

## Performance
- **QwenVL extraction**: ~5-15 seconds per image (depends on API latency)
- **Image preprocessing**: ~100-200ms overhead
- **Claude refinement**: ~10-30 seconds per iteration (if enabled)
- **Agentic loop**: 2-3 iterations typical, ~30-90 seconds total
- **Bottleneck**: API response time, not local compute

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
