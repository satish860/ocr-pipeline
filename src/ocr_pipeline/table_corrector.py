"""
Table Corrector Module

Applies corrections to LaTeX tables based on validation reports.
Supports multiple correction strategies and error types.
"""

import os
import re
import json
import requests
from typing import Dict, List, Optional, Tuple


# ============================================================================
# LaTeX Parsing Utilities
# ============================================================================

def parse_latex_rows(latex: str) -> Tuple[str, List[str], str]:
    r"""
    Parse LaTeX table into header and rows.

    Args:
        latex: Full LaTeX table string

    Returns:
        Tuple of (header_line, data_rows, footer)
        - header_line: \begin{tabular}{...}
        - data_rows: List of row strings (without \\hline)
        - footer: \end{tabular}
    """
    # Extract table header (\begin{tabular}{...})
    header_match = re.match(r'(\\begin\{tabular\}\{[^\}]+\})', latex)
    if not header_match:
        return "", [], ""

    header = header_match.group(1)

    # Extract table footer (\end{tabular})
    footer = r'\end{tabular}'

    # Get content between header and footer
    content_start = len(header)
    content_end = latex.rfind(footer)
    if content_end == -1:
        content_end = len(latex)

    content = latex[content_start:content_end].strip()

    # Split by \\ to get rows, filter out \hline and empty rows
    raw_rows = content.split(r'\\')
    rows = []
    for row in raw_rows:
        row = row.strip()
        # Skip \hline and empty rows
        if row and not row.startswith(r'\hline'):
            rows.append(row)

    return header, rows, footer


def parse_row_cells(row: str) -> List[str]:
    """
    Parse a LaTeX row into individual cells.

    Args:
        row: Row string (without \\)

    Returns:
        List of cell values
    """
    # Split by & (but not escaped \&)
    # Simple approach: split by & and strip whitespace
    cells = row.split('&')
    return [cell.strip() for cell in cells]


def rebuild_row(cells: List[str]) -> str:
    """
    Rebuild a LaTeX row from cells.

    Args:
        cells: List of cell values

    Returns:
        Row string (without \\)
    """
    return ' & '.join(cells)


def rebuild_latex_table(header: str, rows: List[str], footer: str) -> str:
    r"""
    Reconstruct full LaTeX table from components.

    Args:
        header: \begin{tabular}{...}
        rows: List of row strings
        footer: \end{tabular}

    Returns:
        Complete LaTeX table string
    """
    # Join rows with \\ and add \hline between rows
    row_content = ' \\\\\n'.join(rows)
    return f"{header}\n{row_content} \\\\\n{footer}"


def get_cell_value(latex: str, row: int, col: int) -> Optional[str]:
    """
    Extract specific cell value from LaTeX table.

    Args:
        latex: Full LaTeX table string
        row: Row index (0-based)
        col: Column index (0-based)

    Returns:
        Cell value or None if not found
    """
    header, rows, footer = parse_latex_rows(latex)

    if row < 0 or row >= len(rows):
        return None

    cells = parse_row_cells(rows[row])

    if col < 0 or col >= len(cells):
        return None

    return cells[col]


def set_cell_value(latex: str, row: int, col: int, value: str) -> str:
    """
    Replace specific cell value in LaTeX table.

    Args:
        latex: Full LaTeX table string
        row: Row index (0-based)
        col: Column index (0-based)
        value: New value for the cell

    Returns:
        Updated LaTeX table string
    """
    header, rows, footer = parse_latex_rows(latex)

    if row < 0 or row >= len(rows):
        return latex  # Return unchanged if invalid row

    cells = parse_row_cells(rows[row])

    if col < 0 or col >= len(cells):
        return latex  # Return unchanged if invalid col

    # Update cell value
    cells[col] = value

    # Rebuild row
    rows[row] = rebuild_row(cells)

    # Rebuild table
    return rebuild_latex_table(header, rows, footer)


# ============================================================================
# Correction Functions
# ============================================================================

def fix_digit_mismatch(latex: str, error: Dict, image_base64: str) -> Tuple[str, Dict]:
    """
    Fix digit mismatch errors by replacing incorrect value.

    Args:
        latex: LaTeX table string
        error: Error dict with location, latex_value, image_value
        image_base64: Table image (unused for this correction)

    Returns:
        Tuple of (corrected_latex, correction_log)
    """
    row = error['location']['row']
    col = error['location']['column']

    corrected = set_cell_value(latex, row, col, error['image_value'])

    log = {
        "error_type": "digit_mismatch",
        "location": error['location'],
        "original_value": error['latex_value'],
        "corrected_value": error['image_value'],
        "method": "direct_replacement"
    }

    return corrected, log


def fix_spelling(latex: str, error: Dict, image_base64: str) -> Tuple[str, Dict]:
    """
    Fix spelling errors by replacing incorrect value.

    Args:
        latex: LaTeX table string
        error: Error dict with location, latex_value, image_value
        image_base64: Table image (unused for this correction)

    Returns:
        Tuple of (corrected_latex, correction_log)
    """
    row = error['location']['row']
    col = error['location']['column']

    corrected = set_cell_value(latex, row, col, error['image_value'])

    log = {
        "error_type": "spelling",
        "location": error['location'],
        "original_value": error['latex_value'],
        "corrected_value": error['image_value'],
        "method": "direct_replacement"
    }

    return corrected, log


def fix_extra_row(latex: str, error: Dict, image_base64: str) -> Tuple[str, Dict]:
    """
    Fix extra row errors by removing the row.

    Args:
        latex: LaTeX table string
        error: Error dict with location
        image_base64: Table image (unused for this correction)

    Returns:
        Tuple of (corrected_latex, correction_log)
    """
    header, rows, footer = parse_latex_rows(latex)
    row_idx = error['location']['row']

    if 0 <= row_idx < len(rows):
        removed_row = rows.pop(row_idx)
        corrected = rebuild_latex_table(header, rows, footer)

        log = {
            "error_type": "extra_row",
            "location": error['location'],
            "original_value": removed_row,
            "corrected_value": "[row removed]",
            "method": "row_deletion"
        }

        return corrected, log
    else:
        # Invalid row index, return unchanged
        log = {
            "error_type": "extra_row",
            "location": error['location'],
            "original_value": "N/A",
            "corrected_value": "N/A",
            "method": "skipped_invalid_index"
        }
        return latex, log


def fix_missing_row(latex: str, error: Dict, image_base64: str) -> Tuple[str, Dict]:
    """
    Fix missing row errors by re-extracting the row using Gemini.

    Args:
        latex: LaTeX table string
        error: Error dict with location
        image_base64: Table image for re-extraction

    Returns:
        Tuple of (corrected_latex, correction_log)
    """
    header, rows, footer = parse_latex_rows(latex)
    row_idx = error['location']['row']

    # Extract missing row using Gemini
    missing_row_content = extract_missing_row_content(
        image_base64,
        row_idx,
        rows
    )

    if missing_row_content:
        # Insert row at correct position
        rows.insert(row_idx, missing_row_content)
        corrected = rebuild_latex_table(header, rows, footer)

        log = {
            "error_type": "missing_row",
            "location": error['location'],
            "original_value": "[missing]",
            "corrected_value": missing_row_content,
            "method": "gemini_reextraction"
        }

        return corrected, log
    else:
        # Failed to extract, return unchanged
        log = {
            "error_type": "missing_row",
            "location": error['location'],
            "original_value": "[missing]",
            "corrected_value": "N/A",
            "method": "skipped_reextraction_failed"
        }
        return latex, log


def fix_value_swap(latex: str, error: Dict, image_base64: str) -> Tuple[str, Dict]:
    """
    Fix value swap errors by determining swap columns and swapping values.

    Args:
        latex: LaTeX table string
        error: Error dict with location
        image_base64: Table image for swap determination

    Returns:
        Tuple of (corrected_latex, correction_log)
    """
    # Use Gemini to determine which columns to swap
    swap_info = determine_value_swap_columns(image_base64, latex, error)

    if swap_info and 'target_column' in swap_info:
        header, rows, footer = parse_latex_rows(latex)
        row_idx = error['location']['row']
        col_idx = error['location']['column']
        target_col = swap_info['target_column']

        if 0 <= row_idx < len(rows):
            cells = parse_row_cells(rows[row_idx])

            if 0 <= col_idx < len(cells) and 0 <= target_col < len(cells):
                # Swap values
                cells[col_idx], cells[target_col] = cells[target_col], cells[col_idx]
                rows[row_idx] = rebuild_row(cells)
                corrected = rebuild_latex_table(header, rows, footer)

                log = {
                    "error_type": "value_swap",
                    "location": error['location'],
                    "original_value": f"col {col_idx} <-> col {target_col}",
                    "corrected_value": "swapped",
                    "method": "gemini_swap_determination"
                }

                return corrected, log

    # Failed to swap, return unchanged
    log = {
        "error_type": "value_swap",
        "location": error['location'],
        "original_value": "N/A",
        "corrected_value": "N/A",
        "method": "skipped_swap_failed"
    }
    return latex, log


def skip_unclear(latex: str, error: Dict, image_base64: str) -> Tuple[str, Dict]:
    """
    Skip unclear errors (image too blurry to correct).

    Args:
        latex: LaTeX table string
        error: Error dict
        image_base64: Table image (unused)

    Returns:
        Tuple of (unchanged_latex, skip_log)
    """
    log = {
        "error_type": "unclear",
        "location": error['location'],
        "original_value": error.get('latex_value', 'N/A'),
        "corrected_value": "N/A",
        "method": "skipped_unclear"
    }
    return latex, log


# ============================================================================
# Strategy Decision Logic
# ============================================================================

def decide_correction(error: Dict, strategy: str) -> bool:
    """
    Decide whether to correct an error based on strategy.

    Args:
        error: Error dict with confidence and error_type
        strategy: "conservative" | "auto" | "aggressive"

    Returns:
        True if should correct, False if should skip
    """
    confidence = error.get('confidence', 0.0)
    error_type = error.get('error_type', '')

    # Always skip unclear errors
    if error_type == 'unclear':
        return False

    # Strategy-based thresholds
    if strategy == 'conservative':
        return confidence > 0.8
    elif strategy == 'auto':
        return confidence > 0.6
    elif strategy == 'aggressive':
        return confidence > 0.3
    else:
        # Default to conservative
        return confidence > 0.8


# ============================================================================
# Gemini API Helpers
# ============================================================================

def extract_missing_row_content(
    image_base64: str,
    row_index: int,
    context_rows: List[str]
) -> Optional[str]:
    """
    Use Gemini to extract missing row content from table image.

    Args:
        image_base64: Table image
        row_index: Index of missing row
        context_rows: Surrounding rows for context

    Returns:
        Missing row content as LaTeX string, or None if failed
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    # Build context
    context = ""
    if row_index > 0 and row_index - 1 < len(context_rows):
        context += f"Previous row: {context_rows[row_index - 1]}\n"
    if row_index < len(context_rows):
        context += f"Next row: {context_rows[row_index]}\n"

    prompt = f"""Extract the row at position {row_index} from the table image.

Context:
{context}

Return ONLY the row content in LaTeX format (cells separated by &, no \\\\ at end).
Example: Value1 & Value2 & Value3

Return only the row content, nothing else."""

    # Call Gemini API
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
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            }
        ],
        "temperature": 0
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()

        # Extract row content
        row_content = response_data['choices'][0]['message']['content'].strip()

        # Clean up markdown fences if present
        if row_content.startswith("```"):
            row_content = row_content.split("\n", 1)[1]
        if row_content.endswith("```"):
            row_content = row_content.rsplit("\n", 1)[0]

        return row_content.strip()

    except Exception as e:
        print(f"Error extracting missing row: {e}")
        return None


def determine_value_swap_columns(
    image_base64: str,
    latex: str,
    error: Dict
) -> Optional[Dict]:
    """
    Use Gemini to determine which columns should be swapped.

    Args:
        image_base64: Table image
        latex: LaTeX table string
        error: Error dict with location and values

    Returns:
        Dict with 'target_column' key, or None if failed
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    row_idx = error['location']['row']
    col_idx = error['location']['column']

    prompt = f"""A table has a value swap error in row {row_idx}, column {col_idx}.

LaTeX value: {error.get('latex_value', 'N/A')}
Image value: {error.get('image_value', 'N/A')}

Looking at the table image, which column should column {col_idx} swap values with?

Return ONLY a JSON object with format:
{{"target_column": <column_index>}}

Return only the JSON, nothing else."""

    # Call Gemini API
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
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            }
        ],
        "temperature": 0
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()

        # Extract response
        response_text = response_data['choices'][0]['message']['content'].strip()

        # Clean up markdown fences
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Parse JSON
        swap_info = json.loads(response_text)
        return swap_info

    except Exception as e:
        print(f"Error determining value swap: {e}")
        return None


# ============================================================================
# Main Entry Point
# ============================================================================

# Map error types to correction functions
CORRECTION_FUNCTIONS = {
    'digit_mismatch': fix_digit_mismatch,
    'spelling': fix_spelling,
    'extra_row': fix_extra_row,
    'missing_row': fix_missing_row,
    'value_swap': fix_value_swap,
    'unclear': skip_unclear
}


def correct_table(
    latex_table: str,
    validation_report: Dict,
    table_image_base64: str,
    strategy: str = "conservative"
) -> Dict:
    """
    Correct errors in LaTeX table based on validation report.

    Args:
        latex_table: Original LaTeX table
        validation_report: Output from validate_table()
        table_image_base64: Original table image for re-extraction
        strategy: "conservative" | "auto" | "aggressive"

    Returns:
        {
            "corrected_latex": str,
            "corrections_applied": [
                {
                    "error_type": str,
                    "location": dict,
                    "original_value": str,
                    "corrected_value": str,
                    "method": str
                }
            ],
            "corrections_skipped": [
                {
                    "error_type": str,
                    "location": dict,
                    "reason": str
                }
            ]
        }
    """
    # If no errors, return unchanged
    errors = validation_report.get('errors', [])
    if not errors:
        return {
            "corrected_latex": latex_table,
            "corrections_applied": [],
            "corrections_skipped": []
        }

    corrected_latex = latex_table
    corrections_applied = []
    corrections_skipped = []

    # Process each error
    for error in errors:
        error_type = error.get('error_type', 'unknown')

        # Decide if should correct based on strategy
        if not decide_correction(error, strategy):
            corrections_skipped.append({
                "error_type": error_type,
                "location": error.get('location', {}),
                "reason": f"confidence {error.get('confidence', 0):.2f} below threshold for {strategy} strategy"
            })
            continue

        # Get correction function
        correction_func = CORRECTION_FUNCTIONS.get(error_type)

        if correction_func:
            try:
                # Apply correction
                corrected_latex, log = correction_func(
                    corrected_latex,
                    error,
                    table_image_base64
                )

                # Check if correction was actually applied
                if log['method'].startswith('skipped'):
                    corrections_skipped.append({
                        "error_type": error_type,
                        "location": error.get('location', {}),
                        "reason": log['method']
                    })
                else:
                    corrections_applied.append(log)

            except Exception as e:
                corrections_skipped.append({
                    "error_type": error_type,
                    "location": error.get('location', {}),
                    "reason": f"exception: {str(e)}"
                })
        else:
            corrections_skipped.append({
                "error_type": error_type,
                "location": error.get('location', {}),
                "reason": "unknown error type"
            })

    return {
        "corrected_latex": corrected_latex,
        "corrections_applied": corrections_applied,
        "corrections_skipped": corrections_skipped
    }
