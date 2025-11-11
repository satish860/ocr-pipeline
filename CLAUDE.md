# OCR Pipeline Project

## Overview
This project implements a two-stage OCR pipeline:
1. **QwenVL** - Layout detection and bounding box extraction
2. **Mistral AI (via OpenRouter)** - High-quality OCR for each detected region

Output: Formatted Markdown in two styles (document-style and structured)

## Technology Stack
- Python 3.12
- UV for dependency management
- QwenVL (Qwen2.5-VL-7B-Instruct) for layout analysis
- Mistral AI via OpenRouter (pixtral-12b-2409) for OCR
- Pillow for image processing

## Development Philosophy
- Test immediately after implementing each feature
- Incremental development approach
- Run code to verify at each step

## API Setup
- Using OpenRouter for accessing Mistral AI models
- Cost-effective and unified API interface
- Set OPENROUTER_API_KEY in .env file

## Project Structure
```
ocr-pipeline/
├── src/ocr_pipeline/    # Main package
├── examples/            # Usage examples
└── tests/              # Test files
```

## Getting Started
1. Copy `.env.example` to `.env` and add your OpenRouter API key
2. Install dependencies: `uv sync`
3. Run examples: `uv run python examples/basic_usage.py`
