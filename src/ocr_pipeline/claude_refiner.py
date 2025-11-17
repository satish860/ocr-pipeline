"""
Claude Sonnet 4.5 Visual Refinement Module.

This module provides optional visual refinement of QwenVL extraction using
Claude Sonnet 4.5 via OpenRouter API. It follows the OLMoCR-2 approach:

1. QwenVL extracts markdown with all content (text, tables, signatures, handwritten notes, etc.)
2. Python converts LaTeX → HTML (semantic structure)
3. Claude visually verifies and refines ALL content in the HTML
4. Output: Refined semantic HTML document

Claude verifies ALL content types:
- Regular text (paragraphs, headers, labels, captions)
- Tables (structure, cell values, alignment)
- Handwritten elements (signatures, notes, stamps)
- Special elements (charts, images, annotations)
- Text alignment and positioning

Key Design Principles (learned from table_bug.md):
- Conservative refinement: Only fix clear errors
- No text manipulation: Claude re-extracts, we don't parse/modify
- Safety checks: Return original if refinement looks broken
- Transparent: Track when refinement helps vs hurts
"""

import os
import base64
import re
from typing import Dict, Tuple, Optional
from io import BytesIO

import requests
from PIL import Image
from dotenv import load_dotenv

# Import LaTeX to HTML converter
from .latex_to_html import convert_markdown_latex_to_html

# Load environment variables
load_dotenv()


def encode_pil_image_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode PIL Image to base64 string.

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Base64 encoded string
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_claude_refinement_api(
    image_input,  # Can be str (file path) or PIL Image
    html_content: str,
    include_usage: bool = False,
    timeout: int = 120
) -> Tuple[str, Dict]:
    """
    Call OpenRouter API with Claude Sonnet 4.5 to refine HTML extraction.

    This function sends the original document image along with the HTML version
    of QwenVL's extraction to Claude, asking it to verify accuracy and provide
    refinements if needed.

    Args:
        image_input: Either a file path (str) or PIL Image object
        html_content: HTML version of QwenVL extraction (LaTeX→HTML converted)
        include_usage: Whether to include usage/cost data in response
        timeout: Request timeout in seconds

    Returns:
        Tuple of (refined_html, metadata_dict)
        - refined_html: Claude's refined version (or original if no changes)
        - metadata_dict: Contains usage, refinement_applied, error (if any)
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    # Handle both file path and PIL Image
    if isinstance(image_input, str):
        # Load and encode image from file
        with Image.open(image_input) as img:
            base64_image = encode_pil_image_base64(img)
    elif isinstance(image_input, Image.Image):
        # Encode PIL Image directly
        base64_image = encode_pil_image_base64(image_input)
    else:
        raise ValueError("image_input must be either a file path (str) or PIL Image object")

    # Build system message
    system_content = """You are an OCR accuracy verification expert. Your task is to verify and refine ALL content in OCR extraction results.

CRITICAL INSTRUCTIONS:

1. **Visual Verification First**: Compare the provided OCR extraction against the original document image
2. **Conservative Refinement**: Only fix CLEAR errors where the extraction is objectively wrong
3. **Preserve Valuable Information**: If OCR extracted MORE complete text than visible (e.g., "IND/LOR VOLUME" vs just "VOLUME"), that's BETTER, not an error
4. **Trust Good Extractions**: If the extraction looks accurate, return it EXACTLY as provided
5. **No Harmful "Corrections"**: Don't simplify or "clean up" text that is already correct

WHAT TO FIX (ALL CONTENT TYPES):
- **Misread characters or numbers**: Clear OCR errors in any text (paragraphs, headers, labels, etc.)
- **Incorrect table structure**: Wrong columns, missing rows, misaligned data, incorrect cell values
- **Missing content**: Paragraphs, headers, labels, or text completely missed by OCR
- **Wrong text position/order**: Content appearing in wrong reading order
- **Handwritten text errors**: Incorrectly transcribed handwritten notes, signatures, stamps
- **Incorrect alignment**: Text alignment not matching the document (left/center/right)
- **Chart/image descriptions**: Inaccurate or missing descriptions of charts, diagrams, images
- **Special elements**: Incorrectly identified signatures, stamps, or annotations

WHAT NOT TO "FIX":
- More complete text than visible in image (this is good!)
- Abbreviations that are expanded (e.g., "Number" vs "No.")
- Text that matches the image but looks "wrong" to you
- Formatting choices (unless clearly incorrect)
- Minor stylistic differences that don't affect accuracy

OUTPUT FORMAT:
Return refined semantic HTML with:
- **All text content**: Headers, paragraphs, labels, captions
- **Tables**: <table><thead><tr><th>...</th></tr></thead><tbody><tr><td>...</td></tr></tbody></table>
- **Structure**: <div>, <p>, <h1>-<h6> tags for semantic organization
- **Alignment**: <div align="right">, <div align="center"> where appropriate
- **Coordinate annotations**: Preserve <!-- Type (x1, y1, x2, y2) --> for all special elements
- **Handwritten elements**: Accurately transcribe signatures, handwritten notes, stamps
- **If NO changes needed**: Return the EXACT input HTML

Remember: This is an OCR system. Your job is to accurately extract EVERYTHING in the image (text, tables, signatures, handwritten notes, charts), not to "improve" or "correct" the document content itself."""

    # Build user message
    user_content = [
        {
            "type": "text",
            "text": f"""Please verify the accuracy of this OCR extraction against the document image.

Check ALL content types:
- Regular text (paragraphs, headers, labels)
- Tables (structure, cell values, alignment)
- Handwritten elements (signatures, notes, stamps)
- Special elements (charts, images, annotations)
- Text alignment and positioning

If you find clear OCR errors (misread text, wrong structure, missing content, incorrect transcription), provide a refined version.
If the extraction is accurate, return it EXACTLY as provided.

OCR Extraction to Verify (HTML format):

{html_content}"""
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        }
    ]

    # Prepare API request
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "anthropic/claude-sonnet-4.5",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0,  # Deterministic for verification
    }

    # Add usage tracking if requested
    if include_usage:
        payload["usage"] = {"include": True}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()

        # Extract response text
        refined_html = result["choices"][0]["message"]["content"]

        # Clean HTML wrapper if Claude added code fences
        refined_html = clean_html_wrapper(refined_html)

        # Check if Claude actually made changes
        refinement_applied = (refined_html.strip() != html_content.strip())

        # Extract usage data
        usage = result.get("usage", {}) if include_usage else {}

        return refined_html, {
            'success': True,
            'refinement_applied': refinement_applied,
            'usage': usage,
            'error': None
        }

    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_detail = f"\nResponse: {response.text}"
        except:
            pass
        return html_content, {
            'success': False,
            'refinement_applied': False,
            'usage': {},
            'error': f"HTTP error: {str(e)}{error_detail}"
        }

    except Exception as e:
        return html_content, {
            'success': False,
            'refinement_applied': False,
            'usage': {},
            'error': f"API error: {str(e)}"
        }


def clean_html_wrapper(content: str) -> str:
    """
    Remove code fence wrappers from HTML content.

    Claude may return: ```html\\n...\\n```
    We need to strip these for proper rendering.

    Args:
        content: Raw content from API

    Returns:
        Clean HTML content
    """
    # Remove opening ```html or ```markdown code fence
    content = re.sub(r'^```(?:html|markdown)\s*\n', '', content.strip())

    # Remove closing ```
    content = re.sub(r'\n```\s*$', '', content)

    return content.strip()


def validate_refinement(
    original_html: str,
    refined_html: str
) -> Tuple[bool, str]:
    """
    Validate that refinement looks reasonable (safety check).

    This prevents catastrophic failures like the table validation bug where
    refinement completely destroyed the output.

    Args:
        original_html: Original HTML extraction
        refined_html: Claude's refined version

    Returns:
        Tuple of (is_valid, reason)
        - is_valid: True if refinement passes safety checks
        - reason: Explanation if invalid
    """
    # Check 1: Refined output is not empty
    if not refined_html or len(refined_html.strip()) < 10:
        return False, "Refined output is empty or too short"

    # Check 2: Refined output is not drastically shorter (>80% reduction = suspicious)
    original_len = len(original_html)
    refined_len = len(refined_html)

    if refined_len < original_len * 0.2:  # More than 80% reduction
        return False, f"Refined output too short ({refined_len} vs {original_len} chars, {refined_len/original_len:.1%})"

    # Check 3: Key structure markers are preserved
    # Count coordinate annotations
    original_annotations = len(re.findall(r'<!-- \w+ \(\d+, \d+, \d+, \d+\) -->', original_html))
    refined_annotations = len(re.findall(r'<!-- \w+ \(\d+, \d+, \d+, \d+\) -->', refined_html))

    # Allow some variation, but not complete removal
    if original_annotations > 0 and refined_annotations == 0:
        return False, f"All coordinate annotations removed ({original_annotations} -> 0)"

    # Check 4: Table structure preserved if tables exist
    original_tables = len(re.findall(r'<table>', original_html))
    refined_tables = len(re.findall(r'<table>', refined_html))

    if original_tables > 0 and refined_tables == 0:
        return False, f"All tables removed ({original_tables} -> 0)"

    # All checks passed
    return True, "Refinement looks valid"


def remove_inline_base64_images(html: str) -> str:
    """
    Remove inline base64 images from HTML to reduce size for Claude API.

    Inline base64 images are huge and Claude might strip them anyway.
    We keep the structure but remove the base64 data.

    Args:
        html: HTML content with inline base64 images

    Returns:
        HTML with base64 images replaced with placeholders
    """
    # Pattern: ![Type](data:image/png;base64,...)
    # Replace with: [Image: Type]
    pattern = r'!\[([^\]]+)\]\(data:image/[^;]+;base64,[^\)]+\)'
    cleaned = re.sub(pattern, r'[Image: \1]', html)

    return cleaned


def refine_with_claude(
    image_input,  # Can be str (file path) or PIL Image
    qwen_result: Dict,
    include_usage: bool = False
) -> Dict:
    """
    Refine QwenVL extraction using Claude Sonnet 4.5 visual verification.

    This is the main entry point for refinement. It:
    1. Takes QwenVL's markdown extraction (all content: text, tables, signatures, handwritten notes)
    2. Converts LaTeX tables to HTML using Python library
    3. Strips inline base64 images (they're huge and redundant)
    4. Sends HTML structure + original image to Claude for visual verification of ALL content
    5. Returns refined HTML (or original if refinement failed)

    Claude verifies and refines ALL content types:
    - Regular text (paragraphs, headers, labels)
    - Tables (structure, cell values, alignment)
    - Handwritten elements (signatures, notes, stamps)
    - Special elements (charts, images, annotations)
    - Text alignment and positioning

    Args:
        image_input: Either a file path (str) or PIL Image object
        qwen_result: Result dict from extract_document() containing 'markdown', 'images', etc.
        include_usage: Whether to include usage/cost data

    Returns:
        Dict with same structure as qwen_result but with refined HTML:
        - success: bool
        - markdown: refined HTML with ALL content verified (or original HTML if refinement failed validation)
        - images: same as input (not modified)
        - elements: same as input (not modified)
        - refinement: metadata about refinement (applied, usage, error)
    """
    # Extract QwenVL markdown
    qwen_markdown = qwen_result.get('markdown', '')

    if not qwen_markdown:
        # No markdown to refine
        return {
            **qwen_result,
            'refinement': {
                'success': False,
                'refinement_applied': False,
                'error': 'No markdown in qwen_result'
            }
        }

    # Step 1: Convert LaTeX tables to HTML
    print("Converting LaTeX tables to HTML...")
    html_with_images = convert_markdown_latex_to_html(qwen_markdown)

    # Step 2: Strip inline base64 images (they're huge and Claude sees original image anyway)
    html_content = remove_inline_base64_images(html_with_images)

    # Step 2: Send to Claude for refinement
    print("Requesting Claude Sonnet 4.5 visual refinement...")

    refined_html, metadata = call_claude_refinement_api(
        image_input,
        html_content,
        include_usage=include_usage
    )

    if not metadata['success']:
        # API call failed, return original HTML
        print(f"WARNING: Claude API failed: {metadata['error']}")
        print("Returning original HTML (LaTeX->HTML converted)")
        return {
            **qwen_result,
            'markdown': html_content,  # At least return HTML version
            'refinement': metadata
        }

    if not metadata['refinement_applied']:
        # Claude returned same HTML (no changes)
        print("SUCCESS: Claude verified extraction - no changes needed")
        return {
            **qwen_result,
            'markdown': html_content,  # Return HTML version
            'refinement': metadata
        }

    # Claude made changes - validate them
    is_valid, reason = validate_refinement(html_content, refined_html)

    if not is_valid:
        # Refinement failed validation - return original (safety check)
        print(f"WARNING: Refinement failed validation: {reason}")
        print("Returning original HTML (safety rollback)")
        return {
            **qwen_result,
            'markdown': html_content,  # Return original HTML
            'refinement': {
                **metadata,
                'validation_failed': True,
                'validation_reason': reason
            }
        }

    # Refinement is valid - use it
    print("SUCCESS: Claude refinement applied successfully")

    # Return refined result
    return {
        **qwen_result,
        'markdown': refined_html,
        'refinement': {
            **metadata,
            'validation_passed': True
        }
    }
