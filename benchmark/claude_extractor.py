"""Claude Sonnet 4.5 JSON extraction via OpenRouter (with vision + caching)"""

import os
import json
import re
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()


def extract_json_from_markdown(
    markdown: str,
    json_schema: dict,
    extracted_images: Optional[List[Dict]] = None,
    include_usage: bool = False
) -> Dict[str, Any]:
    """
    Extract structured JSON from markdown + images using Claude Sonnet 4.5.

    Args:
        markdown: Markdown text from QwenVL extraction
        json_schema: JSON schema defining expected output structure
        extracted_images: List of dicts with 'base64' and 'type' from QwenVL
        include_usage: Whether to include usage/cost data

    Returns:
        Dict with:
        - success: bool
        - json: extracted JSON (if successful)
        - usage: usage/cost data (if include_usage=True)
        - cache_creation_tokens: tokens written to cache
        - cache_read_tokens: tokens read from cache
        - error: error message (if failed)
        - missing_fields: fields in schema but not extracted
        - extra_fields: fields extracted but not in schema
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    # Clean markdown (remove inline base64 to reduce size, we'll send images separately)
    clean_markdown = remove_inline_base64_images(markdown)

    # Build system message (OpenRouter/OpenAI format - simple string)
    system_content = f"""You are a document data extraction expert. Your task is to extract structured data from documents.

Instructions:
1. **IMPORTANT**: Images are provided for tables, signatures, handwritten text, and other special elements. ALWAYS prioritize reading data from these images over the markdown text, as images contain the original visual data and are more accurate.

2. **For tables - CRITICAL INSTRUCTIONS**:
   - Extract data directly from table images rather than relying on markdown table representations
   - **Identify table structure first**: Look for headers, sections, subsections, and data rows
   - **Nested tables**: If a table has hierarchical levels (main sections → subsections → data rows), map them correctly:
     * Repeating rows with similar structure → Use JSON arrays
     * Unique fields or properties → Use JSON objects
     * Parent-child relationships → Use nested objects/arrays
   - **Merged cells**: If a cell spans multiple rows/columns, extract the value once at the appropriate hierarchical level
   - **Multi-level hierarchies**: Preserve the structure (e.g., category → items → details)
   - **Array vs Object decision**:
     * If the schema shows an array of objects, each table row should become one array element
     * If the schema shows nested objects, map table sections to object properties
   - **Category/classification fields**: Pay close attention to grouping and categorization within tables
   - **Column alignment**: Ensure values are extracted from the correct columns and matched to the right field names

3. **For signatures and handwritten text**: Use the images to read names, dates, and other handwritten information accurately

4. **General extraction rules**:
   - Extract all data from the document that corresponds to fields in the provided JSON schema
   - Return ONLY valid JSON matching the schema structure
   - Use exact field names from the schema
   - If a field is not found in the document, omit it or use null
   - For nested objects and arrays, follow the schema structure exactly
   - Match data types from the schema (strings, numbers, arrays, objects)
   - Preserve formatting when specified (dates, numbers, etc.)
   - No additional text, explanations, or markdown formatting - ONLY the JSON object

JSON Schema:
```json
{json.dumps(json_schema, indent=2)}
```"""

    # Build user message with markdown + images (OpenAI content array format)
    image_count = len(extracted_images) if extracted_images else 0
    user_content = [
        {
            "type": "text",
            "text": f"""Extract data from the following document according to the schema provided in the system message.

{"IMPORTANT: " + str(image_count) + " images are provided below showing tables, signatures, and handwritten text. Prioritize reading data from these images." if image_count > 0 else ""}

Document Content (Markdown):

{clean_markdown}"""
        }
    ]

    # Add images if provided (tables, signatures, handwritten text, etc.)
    if extracted_images:
        for i, img in enumerate(extracted_images, 1):
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img['base64']}"
                }
            })

    # Prepare request
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0,  # Deterministic for data extraction
    }

    # Add usage tracking if requested
    if include_usage:
        payload["usage"] = {"include": True}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        # Extract response text
        content = result["choices"][0]["message"]["content"]

        # Parse JSON from response (may have markdown wrapping)
        extracted_json = parse_json_from_text(content)

        # Validate against schema
        schema_keys = set(json_schema.get('properties', {}).keys())
        extracted_keys = set(extracted_json.keys()) if isinstance(extracted_json, dict) else set()

        missing_fields = list(schema_keys - extracted_keys)
        extra_fields = list(extracted_keys - schema_keys)

        # Extract usage and cache metrics
        usage = result.get('usage', {}) if include_usage else {}
        cache_creation_tokens = usage.get('cache_creation_input_tokens', 0)
        cache_read_tokens = usage.get('cache_read_input_tokens', 0)

        return {
            'success': True,
            'json': extracted_json,
            'usage': usage,
            'cache_creation_tokens': cache_creation_tokens,
            'cache_read_tokens': cache_read_tokens,
            'missing_fields': missing_fields,
            'extra_fields': extra_fields,
            'error': None
        }

    except requests.exceptions.HTTPError as e:
        # Print full error response for debugging
        error_detail = ""
        try:
            error_detail = f"\nResponse: {response.text}"
        except:
            pass
        return {
            'success': False,
            'json': {},
            'usage': {},
            'cache_creation_tokens': 0,
            'cache_read_tokens': 0,
            'error': f"HTTP error: {str(e)}{error_detail}"
        }

    except json.JSONDecodeError as e:
        return {
            'success': False,
            'json': {},
            'usage': {},
            'cache_creation_tokens': 0,
            'cache_read_tokens': 0,
            'error': f"Failed to parse JSON: {str(e)}\nResponse: {content[:500] if 'content' in locals() else 'No response'}"
        }
    except Exception as e:
        return {
            'success': False,
            'json': {},
            'usage': {},
            'cache_creation_tokens': 0,
            'cache_read_tokens': 0,
            'error': f"API error: {str(e)}"
        }


def remove_inline_base64_images(markdown: str) -> str:
    """Remove base64 image data from markdown to reduce size."""
    # Remove ![alt](data:image/...;base64,...)
    pattern = r'!\[.*?\]\(data:image/[^;]+;base64,[^\)]+\)'
    return re.sub(pattern, '[IMAGE]', markdown)


def parse_json_from_text(text: str) -> dict:
    """Parse JSON from text that may have markdown code fences."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except:
        pass

    # Try to extract from code fences
    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # Try to find first { and last }
    start = text.find('{')
    end = text.rfind('}') + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])

    raise json.JSONDecodeError("Could not find valid JSON in response", text, 0)
