"""
JSON Extractor Module

Converts markdown OCR output to structured JSON according to provided schema.
Uses Claude Haiku 4.5 via OpenRouter for extraction.
Supports multimodal extraction (text + images) for improved accuracy.
"""

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class JSONExtractor:
    """Extracts structured JSON from markdown using LLM."""

    def __init__(self, api_key: str = None):
        """
        Initialize JSON extractor.

        Args:
            api_key: OpenRouter API key (optional, reads from env if not provided)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

        self.model = "anthropic/claude-haiku-4.5"

    def extract(
        self,
        markdown: str,
        json_schema: Dict[str, Any],
        image_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured JSON from markdown according to schema.
        Supports multimodal extraction with chart images.

        Args:
            markdown: Markdown text from OCR pipeline (may contain [CHART_IMAGE: filename] placeholders)
            json_schema: JSON schema defining expected structure
            image_dir: Optional directory containing chart images (for multimodal extraction)

        Returns:
            Structured JSON matching the schema

        Raises:
            ValueError: If extraction fails or returns invalid JSON
        """
        # Parse markdown to find chart image references
        chart_images = []
        if image_dir:
            chart_images = self._find_and_load_chart_images(markdown, image_dir)

        # Build system prompt and user message
        system_prompt = self._build_system_prompt(json_schema)
        user_message = self._build_user_message(markdown, chart_images)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=4000
            )

            # Extract JSON from response
            result_text = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if result_text.startswith("```json"):
                result_text = result_text[7:]  # Remove ```json
            elif result_text.startswith("```"):
                result_text = result_text[3:]  # Remove ```

            if result_text.endswith("```"):
                result_text = result_text[:-3]  # Remove trailing ```

            result_text = result_text.strip()

            # Parse JSON
            extracted_json = json.loads(result_text)

            return extracted_json

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {result_text[:200]}")
        except Exception as e:
            raise ValueError(f"JSON extraction failed: {e}")

    def _build_extraction_prompt(self, markdown: str, schema: Dict) -> str:
        """
        Build prompt for JSON extraction.

        Args:
            markdown: Input markdown text
            schema: JSON schema

        Returns:
            Formatted prompt string
        """
        schema_str = json.dumps(schema, indent=2)

        prompt = f"""Extract structured data from the following markdown text according to the provided JSON schema.

**JSON Schema:**
```json
{schema_str}
```

**Markdown Content:**
```markdown
{markdown}
```

**Instructions:**
1. Extract all fields defined in the schema from the markdown content
2. **IMPORTANT - Cross-Section Analysis**: The markdown contains sections separated by `---`. Related data may be split across multiple sections:
   - Look for data in adjacent sections that belong together semantically
   - Combine nearby sections that represent the same entity (e.g., parent categories + detail breakdown)
   - Sections immediately before/after a chart or table often contain related data
3. **Hierarchical Relationships**: When you see:
   - A chart/table with detailed data + separate sections with summary categories → combine into one structure
   - Parent category labels (e.g., "$9.1 BD Medical") + child detail sections → include both levels
   - Geographic/segment breakdowns in multiple places → merge into complete list
4. **Spatial Context Hints**: Pay attention to:
   - Sections marked as "paragraph" near "chart" regions → likely part of the chart
   - Numeric values with labels in separate sections → combine with related tables
   - Text blocks that appear to be category labels → match with corresponding data
5. If a field is not present in the markdown, use null or omit it based on schema requirements
6. Ensure all data types match the schema (strings, numbers, arrays, objects)
7. Preserve the exact structure defined in the schema
8. Return ONLY valid JSON matching the schema structure (no explanations or markdown formatting)

**Example Pattern:**
```
<!-- Region 5: chart -->
<table with 10 detailed items>

<!-- Region 6: paragraph -->
$9.1 Parent Category A

<!-- Region 7: paragraph -->
$4.3 Parent Category B
```
→ Extract: Include BOTH the 10 detailed items AND the 2 parent categories in the same array

Extract the data now:"""

        return prompt

    def _build_system_prompt(self, schema: Dict[str, Any]) -> str:
        """Build system prompt with JSON schema and extraction rules."""
        schema_str = json.dumps(schema, indent=2)

        return f"""You are a JSON extraction specialist. Extract structured data from markdown text and images according to the provided JSON schema.

**JSON Schema:**
```json
{schema_str}
```

**Extraction Rules:**
1. Analyze BOTH the markdown text AND any provided images
2. Cross-reference information between text regions and visual elements
3. For charts/graphs visible in images:
   - Extract ALL data points, labels, and values visible
   - Look for parent categories in nearby text regions
   - Combine hierarchical data (parent + child segments) into single arrays
4. The markdown may contain region markers like `<!-- Region X: type -->` - use these for spatial context
5. Chart placeholders like `[CHART_IMAGE: filename]` indicate visual chart data
6. Extract all fields defined in the schema
7. If a field is not present, use null or omit based on schema requirements
8. Ensure all data types match the schema (strings, numbers, arrays, objects)
9. Preserve the exact structure defined in the schema
10. Return ONLY valid JSON matching the schema structure (no code blocks, no explanations)

**Important for charts with legends/callouts:**
- When you see a chart image WITH nearby text blocks containing dollar amounts or category names
- Those text blocks are PART of the chart data (parent categories or additional segments)
- Include them in the same data array as the chart contents"""

    def _build_user_message(self, markdown: str, chart_images: List[Dict]) -> List[Dict]:
        """Build multimodal user message with text and chart images."""
        content = []

        # Add markdown text first
        content.append({
            "type": "text",
            "text": f"Extract structured data from this document.\n\n**Markdown Content:**\n```markdown\n{markdown}\n```"
        })

        # Add chart images
        for chart in chart_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{chart['base64']}"
                }
            })
            # Add context about which chart this is
            content.append({
                "type": "text",
                "text": f"\n[This is the image for: {chart['filename']}]\n"
            })

        return content

    def _find_and_load_chart_images(self, markdown: str, image_dir: str) -> List[Dict]:
        """
        Find [CHART_IMAGE: filename] placeholders in markdown and load the images.

        Args:
            markdown: Markdown text with chart placeholders
            image_dir: Directory containing chart images

        Returns:
            List of dicts with 'filename' and 'base64' keys
        """
        chart_images = []
        image_dir_path = Path(image_dir)

        # Find all [CHART_IMAGE: filename] placeholders
        pattern = r'\[CHART_IMAGE:\s*([^\]]+)\]'
        matches = re.findall(pattern, markdown)

        print(f"\n[Multimodal] Found {len(matches)} chart image references in markdown")

        for filename in matches:
            filename = filename.strip()
            image_path = image_dir_path / filename

            if image_path.exists():
                try:
                    image = Image.open(image_path)
                    base64_image = self._encode_image_to_base64(image)
                    chart_images.append({
                        'filename': filename,
                        'base64': base64_image
                    })
                    print(f"  [OK] Loaded: {filename}")
                except Exception as e:
                    print(f"  [ERROR] Failed to load {filename}: {e}")
            else:
                print(f"  [ERROR] Not found: {image_path}")

        return chart_images

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Save to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        # Encode to base64
        return base64.b64encode(buffer.read()).decode('utf-8')

    def extract_batch(self, markdown_list: list[str], json_schema: Dict[str, Any]) -> list[Dict[str, Any]]:
        """
        Extract JSON from multiple markdown texts using the same schema.

        Args:
            markdown_list: List of markdown texts
            json_schema: JSON schema to use for all extractions

        Returns:
            List of extracted JSON objects
        """
        results = []
        for i, markdown in enumerate(markdown_list):
            try:
                result = self.extract(markdown, json_schema)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to extract JSON from item {i}: {e}")
                results.append(None)

        return results


def main():
    """Test JSON extractor with a simple example."""
    extractor = JSONExtractor()

    # Test with simple example
    test_markdown = """
    # Employee Information

    - Name: John Doe
    - Age: 30
    - Department: Engineering
    - Salary: $85,000
    """

    test_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "department": {"type": "string"},
            "salary": {"type": "number"}
        },
        "required": ["name", "age"]
    }

    print("Testing JSON Extractor")
    print("=" * 60)
    print("\nInput Markdown:")
    print(test_markdown)
    print("\nJSON Schema:")
    print(json.dumps(test_schema, indent=2))
    print("\nExtracting...")

    try:
        result = extractor.extract(test_markdown, test_schema)
        print("\nExtracted JSON:")
        print(json.dumps(result, indent=2))
        print("\nTest successful!")
    except Exception as e:
        print(f"\nTest failed: {e}")


if __name__ == "__main__":
    main()
