# OCR Pipeline Project

## Overview
This project implements a two-stage OCR pipeline using **OpenRouter APIs**:
1. **QwenVL (via OpenRouter)** - Layout detection and bounding box extraction
2. **Gemini Flash 2.5 (via OpenRouter)** - High-quality OCR for each detected region

**Key Architecture Note**: This is a **lightweight API orchestration** service. Both QwenVL and Gemini run via OpenRouter APIs, meaning **no local GPU/heavy compute is required**. The application only handles image processing and API coordination.

Output: Formatted Markdown with spatial relationship analysis

## Technology Stack
- Python 3.12
- UV for dependency management
- **QwenVL (Qwen2.5-VL-7B-Instruct)** via OpenRouter for layout analysis
- **Gemini Flash 2.5 (google/gemini-2.5-flash)** via OpenRouter for OCR
- FastAPI for REST API
- Pillow for image processing
- PyMuPDF for PDF handling

## Architecture
```
Client → FastAPI → Image Processing → OpenRouter APIs
                                    ├─ QwenVL (layout)
                                    └─ Gemini (OCR)
```

**Compute Requirements**: CPU-only (lightweight). No GPU needed since all ML inference happens via OpenRouter.

## Development Philosophy
- Test immediately after implementing each feature
- Incremental development approach
- Run code to verify at each step

## API Setup
- Using OpenRouter for **both QwenVL and Gemini**
- Cost-effective and unified API interface
- Set `OPENROUTER_API_KEY` in `.env` file

## Project Structure
```
ocr-pipeline/
├── src/
│   ├── ocr_pipeline/       # Core pipeline modules
│   └── api/                # FastAPI application
│       └── main.py         # REST API endpoints
├── input/                  # Place input images here
├── output/                 # Processed results
├── Dockerfile              # Container configuration
└── pyproject.toml          # Dependencies
```

## Getting Started

### Local Development
1. Copy `.env.example` to `.env` and add your OpenRouter API key
2. Install dependencies: `uv sync`
3. Run the API: `uv run uvicorn src.api.main:app --reload`
4. Access API docs: http://localhost:8000/docs
5. Upload images via the `/ocr` endpoint

## API Endpoints

### `POST /ocr`
Process a single image through the OCR pipeline.

**Request**:
- `file`: Image file (PNG, JPG, JPEG)
- `include_annotated_image`: Boolean (optional, default: true)
- `extract_charts_as_tables`: Boolean (optional, default: false) - Convert charts/graphs to markdown tables

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

### Docker
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

### Cloud Platforms
This is a standard Docker container that can be deployed to any platform:
- **Fly.io**, **Railway**, **Render**: Connect GitHub repo
- **Google Cloud Run**: `gcloud run deploy`
- **AWS ECS/Fargate**: Standard Docker deployment

**Required Environment Variable**: `OPENROUTER_API_KEY`

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
- Gemini OCR via OpenRouter: ~$0.XX per region
- No compute costs (CPU-only, lightweight container)
- Storage: Minimal (2 files per page)

## Performance
- **Single image**: ~10-30 seconds (depends on complexity)
- **Batch PDF**: Sequential processing (can be parallelized)
- **Bottleneck**: OpenRouter API latency, not compute

## Current Pipeline Flow (Detailed)

```
Input Image
    ↓
┌─────────────────────────────────────┐
│ ImagePreprocessor (3-stage deskew) │
│ 1. EXIF auto-rotation              │
│ 2. OpenCV heuristic rotation       │ ← ISSUE: Fails on complex forms
│    (detect_orientation method)      │
│ 3. Fine deskew (small angles)      │
└──────────────┬──────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ LayoutDetector (Qwen-30B)           │
│ - Detects elements & bounding boxes │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ RegionExtractor                     │
│ - Crops individual regions          │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ OCRExtractor (Gemini Flash 2.5)     │
│ - Parallel OCR on all regions       │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ SpatialAnalyzer                     │
│ - Analyze spatial relationships     │
└──────────────┬───────────────────────┘
               ↓
         Markdown Output
```

## Features

### Document Classification
The pipeline includes document type detection (form/check/general) with specialized handling:
- **Forms**: Extract as structured HTML with field overlay
- **Checks**: Extract structured JSON with check fields
- **General** (invoices, documents): Markdown output with spatial analysis

### Smart Routing
Automatically routes documents to appropriate extraction pipelines based on document type.
