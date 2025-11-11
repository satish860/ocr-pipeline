# OCR Pipeline Project

## Overview
This project implements a two-stage OCR pipeline using **OpenRouter APIs**:
1. **QwenVL (via OpenRouter)** - Layout detection and bounding box extraction
2. **Mistral AI (via OpenRouter)** - High-quality OCR for each detected region

**Key Architecture Note**: This is a **lightweight API orchestration** service. Both QwenVL and Mistral run via OpenRouter APIs, meaning **no local GPU/heavy compute is required**. The application only handles image processing and API coordination.

Output: Formatted Markdown with spatial relationship analysis

## Technology Stack
- Python 3.12
- UV for dependency management
- **QwenVL (Qwen2.5-VL-7B-Instruct)** via OpenRouter for layout analysis
- **Mistral AI (pixtral-12b-2409)** via OpenRouter for OCR
- FastAPI for REST API
- Pillow for image processing
- PyMuPDF for PDF handling

## Architecture
```
Client → FastAPI → Image Processing → OpenRouter APIs
                                    ├─ QwenVL (layout)
                                    └─ Mistral (OCR)
```

**Compute Requirements**: CPU-only (lightweight). No GPU needed since all ML inference happens via OpenRouter.

## Development Philosophy
- Test immediately after implementing each feature
- Incremental development approach
- Run code to verify at each step

## API Setup
- Using OpenRouter for **both QwenVL and Mistral AI**
- Cost-effective and unified API interface
- Set `OPENROUTER_API_KEY` in `.env` file

## Project Structure
```
ocr-pipeline/
├── src/
│   ├── ocr_pipeline/       # Core pipeline modules
│   └── api/                # FastAPI application
│       └── main.py         # REST API endpoints
├── scripts/
│   ├── pdf_to_images.py    # PDF conversion utility
│   └── batch_process_pdf.py # Batch processing
├── Dockerfile              # Container configuration
├── wrangler.toml           # Cloudflare Containers config
├── test_layout_detector.py # Pipeline testing
└── pyproject.toml          # Dependencies
```

## Getting Started

### Local Development
1. Copy `.env.example` to `.env` and add your OpenRouter API key
2. Install dependencies: `uv sync`
3. Run the API: `uv run uvicorn src.api.main:app --reload`
4. Access docs: http://localhost:8000/docs

### CLI Usage
Process a single image:
```bash
uv run python test_layout_detector.py input/image.png
```

Batch process a PDF:
```bash
uv run python scripts/batch_process_pdf.py "path/to/document.pdf"
```

## API Endpoints

### `POST /ocr`
Process a single image through the OCR pipeline.

**Request**:
- `file`: Image file (PNG, JPG, JPEG)
- `include_annotated_image`: Boolean (optional, default: true)

**Response**:
```json
{
  "success": true,
  "filename": "image.png",
  "detected_elements": 10,
  "rotation_correction_degrees": 0.0,
  "markdown": "# Extracted content...",
  "regions": [
    {
      "index": 1,
      "type": "header",
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "annotated_image": "base64..."
}
```

### `GET /`
Health check endpoint.

## Deployment

### Railway (Recommended - Easiest)
1. Go to https://railway.app
2. Sign up with GitHub
3. New Project → Deploy from GitHub repo → Select `ocr-pipeline`
4. Add environment variable: `OPENROUTER_API_KEY`
5. Railway auto-deploys! ✅

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

### Docker (Local/Any Platform)
1. Build image:
   ```bash
   docker build -t ocr-pipeline .
   ```

2. Run container:
   ```bash
   docker run -p 8000:8000 \
     -e OPENROUTER_API_KEY=your_key \
     ocr-pipeline
   ```

### Alternative Platforms
Since this is a standard Docker container, you can deploy to:
- **Fly.io**: `fly deploy`
- **Railway**: Connect GitHub repo
- **Render**: Connect GitHub repo
- **Google Cloud Run**: `gcloud run deploy`
- **AWS ECS/Fargate**: Standard Docker deployment

## Storage Optimization
The pipeline has been optimized to save only essential outputs:
- **Before**: 20+ files per page (individual regions + markdown)
- **After**: 2 files per page (annotated image + combined markdown)
- **Reduction**: 90% storage savings

## Output Format
Each processed page generates:
1. **Annotated Image** (`page_XXXX_annotated.png`): Visual with bounding boxes
2. **Combined Markdown** (`page_XXXX_complete.md`): Spatially-aware structured text

## Environment Variables
- `OPENROUTER_API_KEY`: Required for API access
- `PORT`: API port (default: 8000)
- `ENVIRONMENT`: production/development

## Cost Considerations
- QwenVL via OpenRouter: ~$0.XX per image
- Mistral OCR via OpenRouter: ~$0.XX per region
- No compute costs (CPU-only, lightweight container)
- Storage: Minimal (2 files per page)

## Performance
- **Single image**: ~10-30 seconds (depends on complexity)
- **Batch PDF**: Sequential processing (can be parallelized)
- **Bottleneck**: OpenRouter API latency, not compute
