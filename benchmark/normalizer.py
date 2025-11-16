"""
Normalization functions for comparing extracted values with ground truth.

Handles format variations in dates, numbers, phone numbers, and other common fields
to reduce false mismatches caused by formatting differences rather than actual errors.
"""

import re
from datetime import datetime
from typing import Any, Optional


# Date pattern matching
DATE_PATTERNS = [
    # ISO 8601 formats
    (r'^(\d{4})-(\d{2})-(\d{2})$', '%Y-%m-%d'),  # 2024-01-15
    (r'^(\d{4})/(\d{2})/(\d{2})$', '%Y/%m/%d'),  # 2024/01/15

    # US formats (MM/DD/YYYY - user preference)
    (r'^(\d{1,2})/(\d{1,2})/(\d{4})$', '%m/%d/%Y'),  # 01/15/2024 or 1/15/2024
    (r'^(\d{1,2})-(\d{1,2})-(\d{4})$', '%m-%d-%Y'),  # 01-15-2024

    # Text formats (with or without comma)
    (r'^(\w+)\s+(\d{1,2}),\s+(\d{4})$', '%B %d, %Y'),  # January 15, 2024
    (r'^(\w+)\s+(\d{1,2}),\s+(\d{4})$', '%b %d, %Y'),  # Jan 15, 2024
    (r'^(\w+)\s+(\d{1,2})\s+(\d{4})$', '%B %d %Y'),    # January 15 2024 (no comma)
    (r'^(\w+)\s+(\d{1,2})\s+(\d{4})$', '%b %d %Y'),    # Jan 15 2024 (no comma)

    # Reversed formats (DD/MM/YYYY - less common, lower priority)
    (r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', '%d.%m.%Y'),  # 15.01.2024 (European)
]

# Currency symbols to strip
CURRENCY_SYMBOLS = r'[$€£¥₹₽¢]'

# Phone number pattern (digits with optional separators)
PHONE_PATTERN = r'^[\d\s\(\)\-\.+]+$'


def normalize_date(value: str) -> Optional[str]:
    """
    Normalize date strings to ISO 8601 format (YYYY-MM-DD).

    Handles various formats:
    - ISO: 2024-01-15
    - US: 01/15/2024, 1/15/2024
    - Text: January 15, 2024, Jan 15, 2024
    - European: 15.01.2024

    Args:
        value: Date string in various formats

    Returns:
        ISO 8601 date string (YYYY-MM-DD) or None if not a valid date
    """
    if not isinstance(value, str):
        return None

    value = value.strip()

    # Try each pattern
    for pattern, date_format in DATE_PATTERNS:
        if re.match(pattern, value):
            try:
                # Parse the date
                dt = datetime.strptime(value, date_format)
                # Return in ISO format
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                # Invalid date (e.g., 13/32/2024), try next pattern
                continue

    return None


def normalize_number(value: str) -> Optional[str]:
    """
    Normalize number strings by removing formatting.

    Handles:
    - Currency symbols: $1,234.56 → 1234.56
    - Thousand separators: 1,234.56 → 1234.56
    - Preserves decimals and negative signs
    - Converts int/float types: 17.0 → "17", 1234 → "1234"

    Args:
        value: Number string with optional currency/formatting, or int/float

    Returns:
        Normalized number string or None if not a number
    """
    # Convert numeric types to string first
    if isinstance(value, (int, float)):
        value = str(value)

    if not isinstance(value, str):
        return None

    # Strip whitespace
    value = value.strip()

    # Remove currency symbols
    value_no_currency = re.sub(CURRENCY_SYMBOLS, '', value)

    # Remove commas (thousand separators)
    value_no_commas = value_no_currency.replace(',', '')

    # Check if it's a valid number (including negatives and decimals)
    number_pattern = r'^-?\d+\.?\d*$'
    if re.match(number_pattern, value_no_commas):
        # Strip unnecessary .0 suffix from whole numbers (17.0 → 17)
        # This handles cases where floats are converted to strings
        if '.' in value_no_commas:
            try:
                # Convert to float and back to remove trailing zeros
                num = float(value_no_commas)
                # If it's a whole number, return without decimal
                if num.is_integer():
                    return str(int(num))
                else:
                    return value_no_commas
            except ValueError:
                # If conversion fails, return as-is
                return value_no_commas
        return value_no_commas

    return None


def normalize_phone(value: str) -> Optional[str]:
    """
    Normalize phone numbers by extracting digits only.

    Handles:
    - (123) 456-7890 → 1234567890
    - +1-123-456-7890 → 11234567890
    - 123.456.7890 → 1234567890

    Args:
        value: Phone number string with optional formatting

    Returns:
        Digits-only string or None if not a phone number
    """
    if not isinstance(value, str):
        return None

    value = value.strip()

    # Check if it looks like a phone number (has digits and common separators)
    if re.match(PHONE_PATTERN, value):
        # Extract only digits
        digits = re.sub(r'\D', '', value)

        # Valid phone number should have at least 7 digits
        if len(digits) >= 7:
            return digits

    return None


def normalize_value(value: Any) -> Any:
    """
    Auto-detect and normalize values (dates, numbers, phone numbers).

    Tries normalization in order:
    1. Dates (most specific patterns - checked first to avoid confusion with phone numbers)
    2. Phone numbers (specific pattern with parentheses/spaces)
    3. Numbers (currency, formatting)
    4. Return original if no pattern matches

    Args:
        value: Value to normalize

    Returns:
        Normalized value or original value if no normalization applies
    """
    # Only normalize strings
    if not isinstance(value, str):
        return value

    # Empty strings return as-is
    if not value.strip():
        return value

    # Try date normalization FIRST (dates like "2024-01-15" can be confused with phone numbers)
    normalized = normalize_date(value)
    if normalized is not None:
        return normalized

    # Try phone number (specific pattern with parentheses or spaces)
    # Only check for phone if it has phone-specific characters (not just dashes)
    if any(char in value for char in ['(', ')', ' ']) and any(c.isdigit() for c in value):
        normalized = normalize_phone(value)
        if normalized is not None:
            return normalized

    # Try number normalization
    normalized = normalize_number(value)
    if normalized is not None:
        return normalized

    # Return original value if no normalization applies
    return value


def normalize_for_comparison(value1: Any, value2: Any) -> tuple[Any, Any]:
    """
    Normalize both values for comparison.

    Ensures both values are normalized the same way before comparison.
    Only normalizes if both values are strings.

    Args:
        value1: First value
        value2: Second value

    Returns:
        Tuple of (normalized_value1, normalized_value2)
    """
    # Only normalize if both are strings
    if isinstance(value1, str) and isinstance(value2, str):
        return (normalize_value(value1), normalize_value(value2))

    # Otherwise return as-is
    return (value1, value2)


def normalize_numeric_comparison(value1: Any, value2: Any) -> tuple[Any, Any]:
    """
    Normalize values for comparison, handling mixed numeric/string types.

    Specifically handles cases like:
    - 17.0 (float) vs "17" (string) → both normalized to "17"
    - "1234.56" (string) vs 1234.56 (float) → both normalized to "1234.56"
    - "$1,234" (string) vs 1234 (int) → both normalized to "1234"

    This fixes the common issue where Claude extracts numbers as floats (17.0)
    but ground truth stores them as strings ("17"), causing false mismatches.

    Args:
        value1: First value (can be int, float, or string)
        value2: Second value (can be int, float, or string)

    Returns:
        Tuple of (normalized_value1, normalized_value2) as strings
    """
    # Convert numeric types to strings first
    if isinstance(value1, (int, float)):
        value1 = str(value1)
    if isinstance(value2, (int, float)):
        value2 = str(value2)

    # Now both should be strings (or other types like None, bool, etc.)
    # Apply standard normalization if both are strings
    if isinstance(value1, str) and isinstance(value2, str):
        return (normalize_value(value1), normalize_value(value2))

    # Otherwise return as-is
    return (value1, value2)


# Utility function for debugging
def explain_normalization(value: str) -> dict:
    """
    Show how a value would be normalized (for debugging).

    Args:
        value: Value to analyze

    Returns:
        Dictionary with normalization details
    """
    result = {
        'original': value,
        'normalized': normalize_value(value),
        'detected_type': None
    }

    # Determine what type was detected
    if normalize_phone(value) is not None:
        result['detected_type'] = 'phone'
        result['phone'] = normalize_phone(value)
    elif normalize_date(value) is not None:
        result['detected_type'] = 'date'
        result['date'] = normalize_date(value)
    elif normalize_number(value) is not None:
        result['detected_type'] = 'number'
        result['number'] = normalize_number(value)
    else:
        result['detected_type'] = 'string (no normalization)'

    return result
