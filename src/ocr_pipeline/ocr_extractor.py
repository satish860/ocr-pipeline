"""OCR text extraction using Gemini Flash 2.5 via OpenRouter"""

import os
import base64
from typing import Union
from pathlib import Path
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv


class OCRExtractor:
    """
    Extracts text from image regions using Gemini Flash 2.5.

    This is the second stage in the pipeline after layout detection.
    Uses context-aware prompting based on element type.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize OCRExtractor with OpenRouter API.

        Args:
            api_key: OpenRouter API key. If None, loads from OPENROUTER_API_KEY env var.
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Provide it via api_key parameter or OPENROUTER_API_KEY env variable."
            )

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        # Use Gemini Flash 2.5 for fast, high-quality OCR
        self.model = "google/gemini-2.5-flash"

    def extract_text(
        self,
        image_input: Union[str, Path, Image.Image],
        element_type: str = "text",
    ) -> str:
        """
        Extract text from an image region with context-aware prompting.

        Args:
            image_input: Path to image file or PIL Image object
            element_type: Type of element (table, paragraph, header, handwritten, etc.)

        Returns:
            Extracted text formatted as Markdown
        """
        # Load and encode image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("image_input must be a file path or PIL Image object")

        # Convert image to base64
        base64_image = self._encode_image_to_base64(image)

        # Generate context-aware prompt
        prompt = self._generate_prompt(element_type)

        # Call OpenRouter API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            extracted_text = response.choices[0].message.content
            return extracted_text.strip()

        except Exception as e:
            raise RuntimeError(f"Error calling OpenRouter API: {e}")

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        import io

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Save to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        # Encode to base64
        return base64.b64encode(buffer.read()).decode('utf-8')

    def _generate_prompt(self, element_type: str) -> str:
        """Generate context-aware OCR prompt based on element type."""

        element_type_lower = element_type.lower()

        if 'table' in element_type_lower:
            return """Extract all text from this table image and format it as a clean Markdown table.

Rules:
- Preserve the exact table structure (rows and columns)
- Use proper Markdown table syntax with | separators
- Include header row if present
- Keep all data exactly as shown
- If cells are empty, use a dash (-)

Return ONLY the markdown table, no additional text or explanation."""

        elif 'handwritten' in element_type_lower:
            return """Extract the handwritten text from this image.

Rules:
- Recognize handwritten text carefully
- Preserve the original language (Hindi, English, etc.)
- Format as clean markdown text
- If multiple lines, preserve line breaks

Return ONLY the extracted text, no additional commentary."""

        elif any(h in element_type_lower for h in ['h1', 'h2', 'h3', 'header', 'heading']):
            return """Extract the heading text from this image.

Rules:
- Format as markdown heading (use # for h1, ## for h2, ### for h3)
- Preserve the exact text
- Return clean markdown heading

Return ONLY the markdown heading, no additional text."""

        elif 'paragraph' in element_type_lower:
            return """Extract the paragraph text from this image.

Rules:
- Extract all visible text
- Preserve line breaks and formatting
- Format as clean markdown text
- Maintain original language

Return ONLY the extracted text, no additional commentary."""

        else:  # Default for unknown types
            return """Extract all visible text from this image and format as clean markdown.

Rules:
- Preserve structure and formatting
- Use appropriate markdown syntax
- Maintain original language
- Keep text exactly as shown

Return ONLY the extracted markdown text, no additional commentary."""
