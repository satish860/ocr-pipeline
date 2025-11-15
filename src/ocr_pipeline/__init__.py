"""OCR Pipeline - QwenVL Document Extraction via OpenRouter"""

__version__ = "0.2.0"

from .qwen_extractor import extract_document

__all__ = [
    "extract_document",
    "__version__"
]
