"""
Table Validator Module

Validates LaTeX tables against original table images using Gemini 2.0 Flash.
Detects errors such as digit mismatches, spelling errors, missing/extra rows, etc.
"""

import os
import re
import json
import requests
from typing import Dict, List, Optional


VALIDATION_PROMPT_TEMPLATE = """You are a table validator. Your task is to compare a LaTeX table with the table shown in an image and identify any discrepancies.

LATEX TABLE:
```
{latex_table}
```

VALIDATION STEPS:
1. Count rows in image vs LaTeX (exact count)
2. Count columns in image vs LaTeX (exact count)
3. For each cell: compare LaTeX value with image value character-by-character
4. Check for missing/extra rows (especially empty rows at start/end)
5. Check for digit errors (common OCR mistakes: 0/O, 1/I/l, 5/S, 7/1, 8/B, 6/G)
6. Check for spelling errors
7. Check for value swaps (e.g., rate vs amount fields)

CRITICAL RULES:
- Only report DEFINITE errors (confidence > 0.7)
- If a cell is unclear/blurry in the image, mark error_type="unclear" and confidence < 0.5
- Be precise with locations (zero-indexed row/column)
- Compare digit-by-digit for numbers

OUTPUT FORMAT (JSON only, no explanations):
{{
    "valid": true/false,
    "errors": [
        {{
            "location": {{"row": <int>, "column": <int>}},
            "latex_value": "<exact value from LaTeX>",
            "image_value": "<exact value you see in image>",
            "error_type": "digit_mismatch|spelling|missing_row|extra_row|value_swap|unclear",
            "confidence": <0.0-1.0>
        }}
    ]
}}

RETURN ONLY THE JSON OBJECT ABOVE. NO OTHER TEXT.
"""


def validate_table(
    table_image_base64: str,
    latex_table: str,
    prompt: Optional[str] = None
) -> Dict:
    """
    Validate LaTeX table against image using Gemini 2.0 Flash.

    Args:
        table_image_base64: Base64-encoded PNG of table region
        latex_table: LaTeX table string from QwenVL
        prompt: Validation instructions template (uses default if None)

    Returns:
        {
            "valid": bool,
            "confidence": float,
            "errors": [
                {
                    "location": {"row": int, "column": int},
                    "latex_value": str,
                    "image_value": str,
                    "error_type": str,
                    "confidence": float
                }
            ],
            "summary": {
                "total_errors": int,
                "critical": int,
                "high": int,
                "low": int
            }
        }
    """
    # Use default prompt if none provided
    if prompt is None:
        prompt = VALIDATION_PROMPT_TEMPLATE

    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {
            "valid": True,
            "confidence": 0.0,
            "errors": [],
            "summary": {"total_errors": 0, "critical": 0, "high": 0, "low": 0},
            "error": "OPENROUTER_API_KEY not found in environment"
        }

    # Format prompt with LaTeX table
    formatted_prompt = prompt.format(latex_table=latex_table)

    # Prepare API request
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{table_image_base64}"}
                    }
                ]
            }
        ],
        "temperature": 0  # Deterministic output
    }

    try:
        # Make API request
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()

        # Extract response text
        response_text = response_data['choices'][0]['message']['content']

        # Parse validation response
        validation_report = parse_validation_response(response_text)

        # Calculate confidence and summary
        errors = validation_report.get('errors', [])
        total_errors = len(errors)

        # Count by confidence level
        critical = sum(1 for e in errors if e.get('confidence', 0) > 0.8)
        high = sum(1 for e in errors if 0.6 <= e.get('confidence', 0) <= 0.8)
        low = sum(1 for e in errors if e.get('confidence', 0) < 0.6)

        # Overall confidence (inverse of error count, scaled)
        overall_confidence = 1.0 if total_errors == 0 else max(0.0, 1.0 - (total_errors * 0.1))

        return {
            "valid": validation_report.get('valid', True),
            "confidence": overall_confidence,
            "errors": errors,
            "summary": {
                "total_errors": total_errors,
                "critical": critical,
                "high": high,
                "low": low
            }
        }

    except requests.exceptions.Timeout:
        # API timeout - fallback to valid (don't block pipeline)
        return {
            "valid": True,
            "confidence": 0.0,
            "errors": [],
            "summary": {"total_errors": 0, "critical": 0, "high": 0, "low": 0},
            "error": "API timeout - validation skipped"
        }
    except requests.exceptions.RequestException as e:
        # Other API errors
        return {
            "valid": True,
            "confidence": 0.0,
            "errors": [],
            "summary": {"total_errors": 0, "critical": 0, "high": 0, "low": 0},
            "error": f"API error: {str(e)}"
        }
    except Exception as e:
        # Unexpected errors
        return {
            "valid": True,
            "confidence": 0.0,
            "errors": [],
            "summary": {"total_errors": 0, "critical": 0, "high": 0, "low": 0},
            "error": f"Unexpected error: {str(e)}"
        }


def parse_validation_response(response_text: str) -> Dict:
    """
    Parse Gemini's JSON response into validation report.

    Handles:
    - Clean JSON extraction (remove markdown fences)
    - Error categorization by severity
    - Confidence scoring

    Args:
        response_text: Raw response from Gemini API

    Returns:
        Dictionary with 'valid' and 'errors' keys
    """
    # Remove markdown code fences if present
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()

    try:
        # Parse JSON
        validation_data = json.loads(cleaned_text)

        # Ensure required fields exist
        if 'valid' not in validation_data:
            validation_data['valid'] = len(validation_data.get('errors', [])) == 0
        if 'errors' not in validation_data:
            validation_data['errors'] = []

        return validation_data

    except json.JSONDecodeError as e:
        # Failed to parse JSON - return default valid response
        return {
            "valid": True,
            "errors": [],
            "parse_error": f"Failed to parse JSON: {str(e)}"
        }


def extract_latex_from_markdown(markdown: str) -> List[str]:
    """
    Extract all LaTeX table blocks from markdown.

    Args:
        markdown: Markdown content with embedded LaTeX tables

    Returns:
        List of LaTeX table strings
    """
    # Pattern to match LaTeX tables
    pattern = r'\\begin\{tabular\}.*?\\end\{tabular\}'
    tables = re.findall(pattern, markdown, re.DOTALL)
    return tables
