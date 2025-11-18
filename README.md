# OCR Pipeline

A lightweight **single-stage OCR pipeline** using **QwenVL (Qwen3-VL-235B)** for end-to-end document extraction with optional Claude Sonnet 4.5 refinement. Both models run via OpenRouter APIs, requiring **no local GPU** - perfect for serverless deployments and cost-effective document processing.

## Features

- **Single-Stage Extraction**: QwenVL (Qwen3-VL-235B) performs complete document extraction in one API call
- **Optional Claude Refinement**:
  - Single-pass refinement with Claude Sonnet 4.5 for improved accuracy
  - Agentic refinement loop with convergence detection for critical documents
- **Image Preprocessing**: Automatic quality analysis, contrast enhancement, sharpening, and intelligent upscaling
- **Markdown Output**: Clean markdown with LaTeX tables, HTML alignment tags, and inline base64 images
- **Coordinate Annotations**: Special elements (tables, charts, signatures, handwritten text) tagged with 0-1000 scale coordinates
- **Image Extraction**: Separate array of extracted images for tables, charts, signatures, and handwritten text
- **Table Conversion**: Optional LaTeX-to-HTML conversion for tables
- **Python API**: Simple `extract_document()` function returns results in memory
- **CLI Interface**: Process images from command line
- **CPU-Only**: Lightweight API orchestration - no GPU required

## Quick Start

### Prerequisites

- Python 3.12+
- [UV](https://github.com/astral-sh/uv) package manager
- OpenRouter API key ([Get one here](https://openrouter.ai/keys))

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ocr-pipeline
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

3. Install dependencies:
```bash
uv sync
```

## Usage

### CLI

Process a single image:
```bash
uv run python -m ocr_pipeline.cli input/document.png

# With preprocessing disabled
uv run python -m ocr_pipeline.cli input/document.png --no-preprocess

# With Claude refinement (single-pass)
uv run python -m ocr_pipeline.cli input/document.png --refine

# With agentic refinement loop (iterative with convergence detection)
uv run python -m ocr_pipeline.cli input/document.png --agentic-refine --max-iterations 3
```

### Python API

```python
from ocr_pipeline import extract_document

# Basic extraction
result = extract_document("path/to/image.png", include_images=True)

# With preprocessing disabled
result = extract_document("path/to/image.png", preprocess=False)

# With Claude refinement
result = extract_document(
    "path/to/image.png",
    refine=True,
    include_usage=True  # Include token usage/cost data
)

# With agentic refinement loop
result = extract_document(
    "path/to/image.png",
    agentic_refine=True,
    max_refinement_iterations=3
)

# Access results
if result['success']:
    # Get markdown (LaTeX tables or HTML if refined)
    markdown = result['markdown']

    # Get extracted images
    for img in result['images']:
        print(f"{img['type']}: {img['bbox']}")
        # img['base64'] contains base64-encoded PNG

    # Get detected elements with coordinates
    for elem in result['elements']:
        print(f"{elem['type']}: {elem['bbox']}")

    # Check quality metrics (if preprocess=True)
    if result.get('quality'):
        print(f"Image quality: {result['quality']}")

    # Check refinement info (if refine=True)
    if result.get('refinement'):
        print(f"Refinement: {result['refinement']}")
else:
    print(f"Error: {result['error']}")
```

## Architecture

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
│ - Convert LaTeX tables to HTML (optional)            │
└──────────────┬───────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────┐
│ Claude Refinement (optional)                         │
│ - Single-pass: Quick accuracy improvement            │
│ - Agentic loop: Iterative refinement with           │
│   convergence detection and quality gates            │
└──────────────┬───────────────────────────────────────┘
               ↓
    Markdown + Extracted Images
```

**Key Point**: Both QwenVL and Claude Sonnet 4.5 run via OpenRouter/Anthropic APIs. This service only handles orchestration and image preprocessing - no heavy ML inference locally.

## Output Format

The `extract_document()` function returns a dictionary with:

```python
{
    "success": True,
    "markdown": "# Document Title\n\n...",  # Markdown with inline base64 images
    "images": [                             # Separate array of extracted images
        {
            "type": "Table",
            "base64": "iVBORw0KGgoAAAANS...",
            "bbox": (100, 200, 800, 600)    # 0-1000 scale
        },
        {
            "type": "Signature",
            "base64": "iVBORw0KGgoAAAANS...",
            "bbox": (50, 850, 300, 950)
        }
    ],
    "elements": [                           # Detected elements with coordinates
        {
            "type": "Table",
            "bbox": (100, 200, 800, 600)
        }
    ],
    "quality": {                            # Image quality metrics (if preprocess=True)
        "sharpness": 245.3,
        "contrast": 67.8,
        "width": 2480,
        "height": 3508,
        "needs_preprocessing": False,
        "preprocessing_applied": False
    },
    "refinement": {                         # Refinement info (if refine=True)
        "iterations": 2,
        "converged": True,
        "final_score": 0.95,
        "iteration_history": [...]
    },
    "usage": {                              # Token usage/cost (if include_usage=True)
        "qwen_tokens": 1234,
        "claude_tokens": 5678,
        "total_cost": 0.0042
    }
}
```

**Note**: Results are returned in memory. No files are saved to disk unless you specify `output_dir` when creating `OCRPipeline` instance.

## Deployment

This is a Python library, not a web service. To deploy:

1. **As a Python Package**: Install via `pip install -e .` or `uv sync`
2. **In Your Application**: Import and use `extract_document()` function
3. **Build Your Own API**: Create FastAPI/Flask wrapper around the library

Example minimal FastAPI wrapper:
```python
from fastapi import FastAPI, File, UploadFile
from ocr_pipeline import extract_document
import tempfile

app = FastAPI()

@app.post("/ocr")
async def process_image(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Process with OCR pipeline
    result = extract_document(tmp_path, include_images=True)

    return result
```

**Required Environment Variable**: `OPENROUTER_API_KEY`

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key (for QwenVL) |
| `ANTHROPIC_API_KEY` | Optional | Your Anthropic API key (for Claude refinement) |

## Project Structure

```
ocr-pipeline/
├── .env.example              # Environment template
├── CLAUDE.md                 # Project documentation for AI assistants
├── README.md                 # This file
├── pyproject.toml            # Dependencies (UV)
├── src/
│   └── ocr_pipeline/         # Core pipeline modules
│       ├── __init__.py       # Package exports (extract_document function)
│       ├── cli.py            # CLI entry point
│       ├── ocr_pipeline.py   # Main pipeline orchestrator
│       ├── image_analyzer.py # Image quality analysis and preprocessing
│       ├── qwen_extractor.py # QwenVL API integration
│       ├── table_converter.py # LaTeX to HTML table conversion
│       └── claude_refiner.py # Claude Sonnet 4.5 refinement (optional)
├── input/                    # Sample input images (optional)
└── output/                   # Output directory (only if output_dir specified)
```

## Technology Stack

- **Python 3.12**: Latest stable Python
- **UV**: Fast, modern package manager
- **QwenVL**: Qwen3-VL-235B-A22B-Instruct (via OpenRouter) for document extraction
- **Claude Sonnet 4.5**: claude-sonnet-4-5 (via Anthropic API) for optional refinement
- **Pillow**: Image manipulation and preprocessing
- **OpenAI Python Library**: For OpenRouter API calls
- **Requests**: HTTP client for API calls

## Performance & Cost

### Performance
- **QwenVL extraction**: ~5-15 seconds per image (depends on API latency)
- **Image preprocessing**: ~100-200ms overhead
- **Claude refinement**: ~10-30 seconds per iteration (if enabled)
- **Agentic loop**: 2-3 iterations typical, ~30-90 seconds total
- **Bottleneck**: API response time, not local compute

### Cost Considerations
- **QwenVL**: ~$0.02-0.05 per image (varies by resolution)
- **Claude refinement**: ~$0.03-0.10 per iteration (varies by content length)
- **Total typical cost**: $0.02-0.15 per document (depending on options)
- **Compute**: Minimal (CPU-only, no GPU needed)
- **Storage**: Results in memory (no disk usage unless `output_dir` specified)

Check [OpenRouter pricing](https://openrouter.ai/docs#models) and [Anthropic pricing](https://www.anthropic.com/pricing) for current rates.

## Development

### Development Philosophy

1. Test immediately after implementing each feature
2. Incremental development approach
3. Run code to verify at each step
4. Exhaust existing code before writing new code

### Testing

Test the CLI with sample images:
```bash
uv run python -m ocr_pipeline.cli input/sample.png
```

Or test the Python API:
```python
from ocr_pipeline import extract_document

result = extract_document("input/sample.png")
print(result['markdown'][:500])  # Print first 500 chars
```

## Troubleshooting

**Error: OPENROUTER_API_KEY not set**
- Copy `.env.example` to `.env`
- Add your OpenRouter API key
- Or set environment variable: `export OPENROUTER_API_KEY=your_key`

**Error: ANTHROPIC_API_KEY not set (when using refinement)**
- Add your Anthropic API key to `.env`
- Or set environment variable: `export ANTHROPIC_API_KEY=your_key`
- Only needed if using `refine=True` or `agentic_refine=True`

**Poor OCR quality**
- Check input image quality (resolution, clarity)
- Try enabling preprocessing if disabled: `preprocess=True` (default)
- Consider using Claude refinement: `refine=True` or `agentic_refine=True`
- Review extracted images to see what QwenVL detected

**Slow performance**
- Normal: 5-15 seconds for QwenVL extraction
- With refinement: 15-45 seconds (single-pass) or 30-90 seconds (agentic)
- Bottleneck is API latency, not your machine

## Key Design Principles

**This is OCR, not data extraction**:
- Goal: Accurately extract what is visible in the image
- NOT our job: "Fix", "correct", or "enhance" document content
- Philosophy: High-fidelity text extraction, not intelligent interpretation

**Refinement Philosophy** (when using Claude):
1. **Aggressive Error Correction**: Fix ALL errors where extraction doesn't match image
   - Don't be conservative - if structure is wrong (especially tables), rebuild completely
   - Table structures frequently need complete reconstruction
2. **Preserve Valuable Information**: Keep text that is MORE complete than visible
   - Expanded abbreviations are good (e.g., "IND/LOR VOLUME" vs just "VOLUME")
   - Don't "correct" the document content itself
3. **Quality Gates Required**: Visual validation, structure validation, automatic rollback

See [CLAUDE.md](CLAUDE.md) for detailed design philosophy and lessons learned.

## Contributing

Contributions welcome! Please:
1. Follow the incremental development approach
2. Test changes immediately
3. Update documentation as needed

## License

TBD

## Links

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [UV Package Manager](https://github.com/astral-sh/uv)
