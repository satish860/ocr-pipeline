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
        output_dir: str = None,
        region_index: int = None,
    ) -> str:
        """
        Extract text from an image region with context-aware prompting.

        Args:
            image_input: Path to image file or PIL Image object
            element_type: Type of element (table, paragraph, header, handwritten, etc.)
            output_dir: Optional directory to save chart images (for Option B testing)
            region_index: Optional region index for naming saved chart images

        Returns:
            Extracted text formatted as Markdown, or image placeholder for charts
        """
        # Load and encode image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("image_input must be a file path or PIL Image object")

        # Option B: For charts, save the image and return a placeholder
        element_type_lower = element_type.lower()
        is_chart = any(chart_type in element_type_lower for chart_type in ['chart', 'graph', 'infographic', 'diagram'])

        if is_chart and output_dir and region_index is not None:
            # Save the chart region image
            os.makedirs(output_dir, exist_ok=True)
            chart_filename = f"chart_region_{region_index}.png"
            chart_path = os.path.join(output_dir, chart_filename)
            image.save(chart_path)
            print(f"  [Option B] Saved chart image: {chart_filename}")

            # Return custom placeholder (easy to parse programmatically)
            return f"[CHART_IMAGE: {chart_filename}]"

        # Normal OCR extraction for non-chart elements or when Option B is not enabled
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
- IMPORTANT: If cells are merged/span multiple rows, repeat the value for each row in the markdown output
- For vertically merged cells, duplicate the cell value for every row it spans

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

        elif any(chart_type in element_type_lower for chart_type in ['chart', 'graph', 'infographic', 'diagram']):
            return """Extract data from this chart/graph/infographic and convert it to a structured table format.

Rules:
- First, identify the chart type (bar chart, pie chart, donut chart, line graph, infographic map, etc.)
- Extract the chart title or heading if present
- Identify all data labels and their corresponding values
- For charts with axes: extract axis labels and all data points
- For pie/donut charts: extract each segment label with its value/percentage
- For infographics/maps: extract each region/element with its associated data
- For charts with legends: match legend items to their data series
- Format the extracted data as an HTML table with proper headers and rows
- Use <table>, <tr>, <td>, <th> tags for structure
- Ensure each data point is matched with its correct label
- If multiple data series exist (e.g., multiple bars per category), create appropriate columns

Format:
1. Start with the chart title as a text line (if present)
2. Follow with an HTML table containing the extracted data
3. Include column headers that describe what each column represents

Example for a pie chart showing "Revenue by Segment":
<table>
  <tr>
    <th>Segment</th>
    <th>Revenue</th>
  </tr>
  <tr>
    <td>Product A</td>
    <td>$5.2B</td>
  </tr>
  <tr>
    <td>Product B</td>
    <td>$3.1B</td>
  </tr>
</table>

Example for a line graph showing "Quarterly Sales":
<table>
  <tr>
    <th>Quarter</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Q1 2024</td>
    <td>145%</td>
  </tr>
  <tr>
    <td>Q2 2024</td>
    <td>156%</td>
  </tr>
</table>

Example for a horizontal bar chart with multiple columns:
<table>
  <tr>
    <th>Category</th>
    <th>Option A</th>
    <th>Option B</th>
    <th>Option C</th>
  </tr>
  <tr>
    <td>Item 1</td>
    <td>45%</td>
    <td>35%</td>
    <td>20%</td>
  </tr>
</table>

Return ONLY the title (if present) and HTML table, no additional commentary or explanation."""

        else:  # Default for unknown types
            return """Extract all visible text from this image and format as clean markdown.

Rules:
- Preserve structure and formatting
- Use appropriate markdown syntax
- Maintain original language
- Keep text exactly as shown

Return ONLY the extracted markdown text, no additional commentary."""
