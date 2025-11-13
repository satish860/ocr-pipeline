"""
JSON Extractor Module

Converts markdown OCR output to structured JSON according to provided schema.
Uses Claude Haiku 4.5 via OpenRouter for extraction.
"""

import json
import os
from typing import Any, Dict
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

        self.model = "anthropic/claude-3.5-haiku"

    def extract(self, markdown: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured JSON from markdown according to schema.

        Args:
            markdown: Markdown text from OCR pipeline
            json_schema: JSON schema defining expected structure

        Returns:
            Structured JSON matching the schema

        Raises:
            ValueError: If extraction fails or returns invalid JSON
        """
        prompt = self._build_extraction_prompt(markdown, json_schema)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
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

Instructions:
1. Extract all fields defined in the schema from the markdown content
2. If a field is not present in the markdown, use null or omit it based on schema requirements
3. Ensure all data types match the schema (strings, numbers, arrays, objects)
4. Preserve the exact structure defined in the schema
5. Return ONLY valid JSON matching the schema structure (no explanations or markdown formatting)

Extract the data now:"""

        return prompt

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
