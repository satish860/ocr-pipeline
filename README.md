# OCR Pipeline

A lightweight two-stage OCR pipeline that combines **QwenVL** for intelligent layout detection and **Gemini Flash 2.5** for high-quality text extraction. Both models run via OpenRouter APIs, requiring **no local GPU** - perfect for serverless deployments.

## Features

- **Two-Stage Pipeline**:
  - **Stage 1**: QwenVL (Qwen2.5-VL-7B-Instruct) detects document layout and extracts bounding boxes
  - **Stage 2**: Gemini Flash 2.5 (google/gemini-2.5-flash) performs OCR on each detected region
- **Image Preprocessing**: Automatic deskewing and rotation correction
- **Layout Detection**: Identifies tables, paragraphs, headers, and handwritten text
- **Spatial Analysis**: Maintains spatial relationships between text regions
- **Document Classification**: Intelligent routing for forms, checks, and general documents
- **Structured Extraction**: JSON extraction for forms and checks, markdown for documents
- **REST API**: Production-ready FastAPI endpoints
- **Storage Optimized**: Saves only annotated images and combined markdown (90% storage reduction)
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

### REST API

Start the FastAPI server:
```bash
uv run uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

#### API Endpoints

**Health Check**
```bash
GET /
```

**Process Image**
```bash
POST /ocr
```

**Request Parameters:**
- `file`: Image file (PNG, JPG, JPEG)
- `include_annotated_image`: Boolean (optional, default: true)
- `extract_charts_as_tables`: Boolean (optional, default: false)

**Response Example:**
```json
{
  "success": true,
  "filename": "invoice.png",
  "detected_elements": 15,
  "rotation_correction_degrees": 2.3,
  "markdown": "# Invoice\n\n**Company Name**\nAddress Line 1...",
  "regions": [
    {
      "index": 1,
      "type": "header",
      "bbox": [50, 30, 450, 80],
      "text": "Invoice"
    },
    {
      "index": 2,
      "type": "table",
      "bbox": [50, 100, 550, 400],
      "text": "| Item | Qty | Price |\n|------|-----|-------|..."
    }
  ],
  "annotated_image": "data:image/png;base64,iVBORw0KG..."
}
```

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/ocr" \
  -F "file=@invoice.png" \
  -F "include_annotated_image=true"
```

Output files will be saved to the `output/` directory:
- `*_annotated.png` - Visual with colored bounding boxes
- `*_complete.md` - Markdown with spatial relationships

## Architecture

```
┌─────────┐     ┌──────────────┐     ┌─────────────────┐
│  Client │────▶│   FastAPI    │────▶│ Image Processor │
└─────────┘     └──────────────┘     └─────────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
                ┌─────────────┐                            ┌──────────────┐
                │  OpenRouter │                            │  OpenRouter  │
                │   (QwenVL)  │                            │   (Gemini)   │
                │   Layout    │────▶ Bounding Boxes ──────▶│     OCR      │
                └─────────────┘                            └──────────────┘
                                                                   │
                                                                   ▼
                                                           ┌───────────────┐
                                                           │   Markdown    │
                                                           │  + Annotated  │
                                                           │     Image     │
                                                           └───────────────┘
```

**Key Point**: Both QwenVL and Gemini Flash 2.5 run via OpenRouter APIs. This service only handles orchestration and image preprocessing - no heavy ML inference locally.

## Output Format

Each processed image/page generates two files:

1. **Annotated Image** (`*_annotated.png`):
   - Original image with colored bounding boxes
   - Color-coded by region type:
     - Red: Headers
     - Blue: Paragraphs
     - Green: Tables
     - Purple: Handwritten text

2. **Combined Markdown** (`*_complete.md`):
   - Spatially-aware text extraction
   - Structured by region type
   - Maintains reading order and relationships
   - Includes metadata (region count, rotation correction)

## Deployment

### Docker

Build and run locally:
```bash
docker build -t ocr-pipeline .
docker run -p 8000:8000 -e OPENROUTER_API_KEY=your_key ocr-pipeline
```

### Cloud Platforms

Deploy to any container platform:
- **Railway**, **Render.com**: Connect GitHub repo, auto-deploy from Dockerfile
- **Fly.io**: `fly deploy`
- **Google Cloud Run**: `gcloud run deploy`
- **AWS ECS/Fargate**: Standard Docker deployment

**Required Environment Variable**: `OPENROUTER_API_KEY`

## Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key | - |
| `PORT` | No | API server port | 8000 |
| `ENVIRONMENT` | No | Environment mode | production |

## Project Structure

```
ocr-pipeline/
├── .env.example              # Environment template
├── CLAUDE.md                 # AI assistant documentation
├── README.md                 # This file
├── Dockerfile                # Container configuration
├── pyproject.toml            # Dependencies (UV)
├── src/
│   ├── ocr_pipeline/         # Core pipeline modules
│   │   ├── __init__.py
│   │   ├── cli.py            # CLI entry point
│   │   ├── image_preprocessor.py    # Deskewing & rotation
│   │   ├── layout_detector.py       # QwenVL integration
│   │   ├── ocr_extractor.py         # Gemini OCR
│   │   ├── region_extractor.py      # Bounding box processing
│   │   ├── spatial_analyzer.py      # Spatial relationships
│   │   ├── document_classifier.py   # Document type detection
│   │   ├── json_extractor.py        # Structured data extraction
│   │   └── smart_router.py          # Routing logic
│   └── api/
│       ├── __init__.py
│       └── main.py           # FastAPI REST endpoints
├── input/                    # Input files directory
└── output/                   # Generated files directory
```

## Technology Stack

- **Python 3.12**: Latest stable Python
- **UV**: Fast, modern package manager
- **FastAPI**: High-performance REST API framework
- **QwenVL**: Qwen2.5-VL-7B-Instruct (via OpenRouter) for layout analysis
- **Gemini Flash 2.5**: google/gemini-2.5-flash (via OpenRouter) for OCR
- **Pillow**: Image manipulation
- **OpenCV**: Image preprocessing (deskewing, rotation)
- **PyMuPDF**: PDF processing
- **BeautifulSoup4**: HTML/XML parsing

## Performance & Cost

### Performance
- **Single image**: ~10-30 seconds (varies by complexity and API latency)
- **Batch PDF**: Sequential processing (parallelizable)
- **Bottleneck**: OpenRouter API response time, not compute

### Cost Considerations
- **QwenVL** (layout detection): ~$0.XX per image
- **Gemini OCR**: ~$0.XX per region
- **Compute**: Minimal (CPU-only, lightweight container)
- **Storage**: 2 files per page (annotated image + markdown)

Check [OpenRouter pricing](https://openrouter.ai/docs#models) for current rates.

## Storage Optimization

This pipeline is optimized for minimal storage:
- **Before**: 20+ files per page (individual regions + metadata)
- **After**: 2 files per page (annotated image + combined markdown)
- **Savings**: ~90% reduction in file count

## Development

### Development Philosophy

1. Test immediately after implementing each feature
2. Incremental development approach
3. Run code to verify at each step

### Testing

Start the development server:
```bash
uv run uvicorn src.api.main:app --reload
```

Then use the interactive API docs at `http://localhost:8000/docs` or test with curl.

## Troubleshooting

**Error: README.md not found during Docker build**
- The Dockerfile expects a README.md in the project root
- Ensure this file exists before building

**Error: OPENROUTER_API_KEY not set**
- Copy `.env.example` to `.env`
- Add your OpenRouter API key
- For Docker: Pass as `-e OPENROUTER_API_KEY=your_key`

**Poor OCR quality**
- Check input image quality (resolution, clarity)
- Verify proper deskewing (check rotation_correction_degrees in output)
- Review annotated image to confirm bounding boxes are accurate

## Contributing

Contributions welcome! Please:
1. Follow the incremental development approach
2. Test changes immediately
3. Update documentation as needed

## License

TBD

## Links

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Railway Deployment](https://railway.app)
- [UV Package Manager](https://github.com/astral-sh/uv)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
