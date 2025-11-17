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


def encode_pil_image_base64(image: Image.Image, format: str = "PNG", max_size_mb: float = 4.0) -> str:
    """
    Encode PIL Image to base64 string, resizing if needed to stay under size limit.

    Claude API has a 5 MB limit for base64 images. We target 4.0 MB to have safety margin.

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
        max_size_mb: Maximum size in MB (default: 4.5 MB for safety margin)

    Returns:
        Base64 encoded string
    """
    # Try encoding at full size first
    buffer = BytesIO()
    image.save(buffer, format=format)
    binary_data = buffer.getvalue()
    encoded = base64.b64encode(binary_data).decode("utf-8")

    # Calculate size in MB
    # Base64 encoding adds ~33% overhead, so len(encoded) ≈ len(binary) * 1.33
    # But we care about the binary size that Claude receives
    binary_size_mb = len(binary_data) / (1024 * 1024)

    if binary_size_mb <= max_size_mb:
        return encoded

    # Image too large - resize iteratively
    print(f"  Image too large ({binary_size_mb:.1f} MB), resizing to fit {max_size_mb} MB limit...")

    # Calculate scale factor needed
    # Size is roughly proportional to pixel count (width * height)
    scale_factor = (max_size_mb / binary_size_mb) ** 0.5  # Square root because 2D
    scale_factor *= 0.85  # Add safety margin (more conservative)

    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)

    print(f"  Resizing from {image.width}x{image.height} to {new_width}x{new_height}")

    # Resize and re-encode
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    resized.save(buffer, format=format)
    binary_data = buffer.getvalue()
    encoded = base64.b64encode(binary_data).decode("utf-8")

    # Verify new size
    new_binary_size_mb = len(binary_data) / (1024 * 1024)
    print(f"  Resized image size: {new_binary_size_mb:.1f} MB")

    return encoded


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
    system_content = """You are an OCR accuracy verification and refinement expert. Your task is to aggressively improve OCR extraction quality by fixing ALL errors.

CRITICAL INSTRUCTIONS:

1. **Visual Verification First**: Compare the provided OCR extraction against the original document image
2. **Aggressive Refinement**: Fix ALL errors where the extraction doesn't match the image - don't be conservative
3. **Preserve Valuable Information**: If OCR extracted MORE complete text than visible (e.g., "IND/LOR VOLUME" vs just "VOLUME"), that's BETTER, keep it
4. **Rebuild When Necessary**: If structure is wrong (especially tables), rebuild it completely from the image
5. **Text Accuracy**: Don't "correct" the document content itself, just extract what's actually there

WHAT TO FIX (ALL CONTENT TYPES):
- **Misread characters or numbers**: Any OCR errors in text (paragraphs, headers, labels, etc.)
- **Incorrect table structure**: Wrong columns, missing rows, misaligned data, incorrect cell values
- **Missing content**: Paragraphs, headers, labels, or text completely missed by OCR
- **Wrong text position/order**: Content appearing in wrong reading order
- **Handwritten text errors**: Incorrectly transcribed handwritten notes, signatures, stamps
- **Incorrect alignment**: Text alignment not matching the document (left/center/right)
- **Chart/image descriptions**: Inaccurate or missing descriptions of charts, diagrams, images
- **Special elements**: Incorrectly identified signatures, stamps, or annotations

CRITICAL TABLE STRUCTURE VERIFICATION:

Tables are the #1 source of OCR errors. Follow these steps:

**Step 0: DETECT TABLE BOUNDARIES**
- Look at the image to identify table borders (usually black lines forming rectangles)
- If the OCR extracted multiple separate tables BUT they are all within ONE continuous outer border → they should be ONE unified table
- If tables are visually separated with space between them → keep them separate
- Common mistake: OCR splits a single invoice/form table into multiple tables based on internal sections

**Step 1: COUNT COLUMNS**
- Look at the image and count the vertical gridlines/column separators for the ENTIRE table
- This is the TRUE column count - ignore what the OCR says
- Different rows may use colspan to span multiple columns
- Example: If you see 12 vertical separators across the full table width, the table has 12 columns

**Step 2: IDENTIFY HEADER ROWS**
- Check if row 1 has different styling (bold text, background color, larger font)
- If row 1 looks like regular data cells → it's NOT a header row, use <tbody> for all rows
- Only use <thead> if the row is visually distinct as a header

**Step 3: DETECT MERGED CELLS**
- Look for cells that span multiple columns (colspan) or rows (rowspan)
- Count the cell borders carefully to determine exact span
- Example: Cell with text spanning 3 column widths → colspan="3"

**Step 4: VALIDATE EACH ROW**
- Count cells in each row: regular cells + (colspan values) + (rowspan values) must equal total columns
- Example: Table with 12 columns, row has 3 cells:
  - Cell 1: colspan="6" (6 columns)
  - Cell 2: colspan="4" (4 columns)
  - Cell 3: regular cell (1 column)
  - Cell 4: regular cell (1 column)
  - Total: 6 + 4 + 1 + 1 = 12 ✓

**Step 5: REBUILD IF WRONG**
- If the OCR table structure is incorrect, COMPLETELY REBUILD it from scratch
- Don't try to "patch" a broken structure - start fresh based on visual analysis
- Extract cell contents into the correct structure you determined in Steps 1-4

**Example 1: Merging Split Tables**
```
OCR extracted 3 separate tables, but image shows ONE continuous bordered table:
❌ WRONG: <table>...</table> ... <table>...</table> ... <table>...</table>
✓ CORRECT: Merge into ONE table with multiple sections using colspan for varying row structures

<table>
<tbody>
  <!-- First section: 3 columns -->
  <tr>
    <td colspan="4">GSTIN: ...</td>
    <td colspan="6">Voucher NO: ...</td>
    <td colspan="2">Date:</td>
  </tr>

  <!-- Section header spanning full width -->
  <tr>
    <td colspan="12"><strong>Details of Service Receiver</strong></td>
  </tr>

  <!-- Main data section: 12 columns -->
  <tr>
    <td>S.No.</td>
    <td>ITS</td>
    <td>SAC Code</td>
    ...12 columns total...
  </tr>
</tbody>
</table>
```

**Example 2: Table Row Validation**
```
Image shows 12-column table:
Row with text spanning columns → use colspan

Correct:
  <td colspan="6">Invoice Value - Nine Lac...</td>
  <td colspan="4"></td>
  <td>Invoice Total</td>
  <td>9,04,116</td>
Total: 6 + 4 + 1 + 1 = 12 ✓
```

WHAT NOT TO "FIX":
- More complete text than visible in image (this is good!)
- Abbreviations that are expanded (e.g., "Number" vs "No.")
- Don't "correct" the document content itself (if invoice says "9,04,116" don't change it to "904116")

OUTPUT FORMAT:
Return refined semantic HTML with:
- **All text content**: Headers, paragraphs, labels, captions
- **Tables**: <table><thead><tr><th>...</th></tr></thead><tbody><tr><td>...</td></tr></tbody></table>
- **Structure**: <div>, <p>, <h1>-<h6> tags for semantic organization
- **Alignment**: <div align="right">, <div align="center"> where appropriate
- **Coordinate annotations**: Preserve <!-- Type (x1, y1, x2, y2) --> for all special elements
- **Handwritten elements**: Accurately transcribe signatures, handwritten notes, stamps

CRITICAL OUTPUT RULE:
Return ONLY the refined HTML/Markdown content itself. Do NOT include:
- Explanations of what you changed
- Preambles like "I need to verify..." or "Here is the corrected version:"
- Issue lists or reasoning
- Any text before or after the actual content

Start your response IMMEDIATELY with the first line of the refined document (e.g., <div align="center">...).

Remember: Your job is to accurately extract EVERYTHING in the image (text, tables, signatures, handwritten notes, charts) and FIX ALL structural and content errors. Be aggressive about fixing tables - they are usually wrong!"""

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

        # Check for error response (e.g., image size limit exceeded)
        if "error" in result:
            error_data = result["error"]
            # Try to extract the real error from nested metadata
            if isinstance(error_data, dict):
                # Check for nested provider error in metadata
                metadata = error_data.get("metadata", {})
                if "raw" in metadata:
                    # OpenRouter wraps provider errors - try to parse them
                    try:
                        import json
                        raw_error = json.loads(metadata["raw"])
                        if "error" in raw_error and "message" in raw_error["error"]:
                            error_msg = raw_error["error"]["message"]
                        else:
                            error_msg = error_data.get("message", str(error_data))
                    except:
                        error_msg = error_data.get("message", str(error_data))
                else:
                    error_msg = error_data.get("message", str(error_data))
            else:
                error_msg = str(error_data)

            return html_content, {
                'success': False,
                'refinement_applied': False,
                'usage': {},
                'error': f"API error: {error_msg}"
            }

        # Extract response text
        if "choices" not in result:
            return html_content, {
                'success': False,
                'refinement_applied': False,
                'usage': {},
                'error': f"Unexpected API response format. Keys: {list(result.keys())}"
            }

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


def refine_with_agentic_loop(
    image_input,  # Can be str (file path) or PIL Image
    qwen_result: Dict,
    max_iterations: int = 2,
    include_usage: bool = False
) -> Dict:
    """
    Agentic refinement loop - Claude iteratively refines its own output.

    Iterative refinement with smart stopping:
    - Iteration 1: Refine QwenVL extraction
    - Iteration N: Review and refine previous iteration
    - Stops early when converged (no changes detected)
    - Respects max_iterations safety limit

    Args:
        image_input: Either a file path (str) or PIL Image object
        qwen_result: Result dict from extract_document()
        max_iterations: Maximum refinement iterations (default: 2)
        include_usage: Whether to include usage/cost data

    Returns:
        Dict with refined content + iteration history + stopping reason
    """
    # Extract QwenVL markdown
    qwen_markdown = qwen_result.get('markdown', '')

    if not qwen_markdown:
        return {
            **qwen_result,
            'refinement': {
                'success': False,
                'refinement_applied': False,
                'error': 'No markdown in qwen_result'
            }
        }

    # Convert LaTeX to HTML once (before iterations)
    print("Converting LaTeX tables to HTML...")
    html_with_images = convert_markdown_latex_to_html(qwen_markdown)
    html_content = remove_inline_base64_images(html_with_images)

    # Track iteration history and stopping reason
    iterations = []
    current_html = html_content
    stopping_reason = "max_iterations_reached"

    # Iteration loop
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"Refinement Iteration {iteration}/{max_iterations}")
        print(f"{'='*60}")

        # Build prompt for this iteration
        if iteration == 1:
            # First iteration: standard refinement
            prompt_html = current_html
            print("Task: Refine QwenVL extraction")
        else:
            # Subsequent iterations: review previous refinement
            prompt_html = current_html
            print("Task: Review and refine previous iteration")

        # Call Claude
        refined_html, metadata = call_claude_refinement_api(
            image_input,
            prompt_html,
            include_usage=include_usage
        )

        if not metadata['success']:
            print(f"ERROR: Claude API failed: {metadata['error']}")
            print("Stopping iterations early")
            stopping_reason = "api_error"
            break

        # Validate refinement
        is_valid, reason = validate_refinement(current_html, refined_html)

        if not is_valid:
            print(f"WARNING: Refinement failed validation: {reason}")
            print("Stopping iterations early")
            iterations.append({
                'iteration': iteration,
                'html': current_html,
                'changes_made': False,
                'validation_failed': True,
                'validation_reason': reason,
                'usage': metadata.get('usage', {})
            })
            stopping_reason = "validation_failed"
            break

        # Check if anything changed
        changes_made = refined_html.strip() != current_html.strip()

        # Store iteration result
        iterations.append({
            'iteration': iteration,
            'html': refined_html,
            'changes_made': changes_made,
            'validation_passed': True,
            'usage': metadata.get('usage', {})
        })

        # Print iteration summary
        if changes_made:
            print(f"[CHANGE] Changes detected (length: {len(current_html)} -> {len(refined_html)} chars)")
        else:
            print("[NO CHANGE] Output identical to input")

        # Update current HTML for next iteration
        current_html = refined_html

        # Phase 2: Early stopping if converged
        if not changes_made:
            print(f"\nConverged: No changes detected in iteration {iteration}")
            print("Stopping early to save API costs")
            stopping_reason = "converged"
            break

    # Build final result
    print(f"\n{'='*60}")
    print(f"Agentic Refinement Complete: {len(iterations)} iterations")
    print(f"{'='*60}")

    # Calculate total usage
    total_usage = {}
    if include_usage:
        for iter_data in iterations:
            usage = iter_data.get('usage', {})
            for key, value in usage.items():
                total_usage[key] = total_usage.get(key, 0) + value

    return {
        **qwen_result,
        'markdown': current_html,  # Final refined HTML
        'refinement': {
            'success': True,
            'agentic': True,
            'iterations': len(iterations),
            'iteration_history': iterations,
            'total_usage': total_usage,
            'final_changes_made': iterations[-1]['changes_made'] if iterations else False,
            'stopping_reason': stopping_reason
        }
    }
