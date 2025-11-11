# OCR Pipeline

A two-stage OCR pipeline combining QwenVL for layout detection and Mistral AI (via OpenRouter) for high-quality text extraction.

## Features

- **Stage 1**: QwenVL detects document layout and extracts bounding boxes
- **Stage 2**: Mistral AI performs OCR on each detected region
- **Output**: Formatted Markdown in both document-style and structured formats

## Setup

### Prerequisites

- Python 3.12+
- [UV](https://github.com/astral-sh/uv) package manager
- OpenRouter API key ([Get one here](https://openrouter.ai/keys))

### Installation

1. Clone or navigate to the project directory:
```bash
cd ocr-pipeline
```

2. Copy the environment template and add your API key:
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

3. Install dependencies:
```bash
uv sync
```

### Verify Installation

Run the hello world command:
```bash
uv run ocr-pipeline
```

You should see:
```
Hello from OCR Pipeline!
Using QwenVL + Mistral AI via OpenRouter
Version: 0.1.0
```

## Project Structure

```
ocr-pipeline/
├── .env.example          # Environment template
├── .gitignore           # Git ignore rules
├── CLAUDE.md            # Project documentation for AI assistants
├── README.md            # This file
├── pyproject.toml       # UV project configuration
├── src/
│   └── ocr_pipeline/    # Main package
│       ├── __init__.py
│       └── cli.py       # CLI entry point
├── examples/            # (Coming soon) Usage examples
└── tests/              # (Coming soon) Test files
```

## Technology Stack

- **Python 3.12**: Latest stable Python
- **UV**: Fast Python package manager
- **QwenVL**: Qwen2.5-VL-7B-Instruct for layout analysis
- **Mistral AI**: pixtral-12b-2409 for OCR (via OpenRouter)
- **Pillow**: Image processing

## Next Steps

- Implement QwenVL layout detector
- Implement Mistral AI OCR engine
- Create pipeline orchestration
- Add Markdown formatters
- Build CLI commands

## Development

Follow the incremental development approach:
1. Implement one feature at a time
2. Test immediately after implementation
3. Verify functionality before moving forward

## License

TBD
