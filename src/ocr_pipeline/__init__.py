"""OCR Pipeline - QwenVL Document Extraction via OpenRouter"""

__version__ = "0.2.0"

# Main pipeline orchestrator (recommended entry point)
from .ocr_pipeline import OCRPipeline

# Individual components (for advanced usage)
from .image_analyzer import ImageAnalyzer
from .qwen_extractor import QwenExtractor
from .table_converter import TableConverter

__all__ = [
    "OCRPipeline",
    "ImageAnalyzer",
    "QwenExtractor",
    "TableConverter",
    "__version__"
]
