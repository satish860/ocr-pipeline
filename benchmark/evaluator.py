"""
Evaluation metrics for OCR benchmark using strict matching (getomni-ai methodology).

Implements the official benchmark evaluation:
Accuracy = 1 - (number of different fields / total fields)
"""

from typing import Dict, List, Any, Tuple


def compare_json(
    predicted: Any,
    ground_truth: Any,
    path: str = ""
) -> Dict[str, Any]:
    """
    Recursively compare predicted and ground truth JSON using strict matching.

    Implements getomni-ai benchmark methodology:
    - Count ALL fields recursively (including nested objects and arrays)
    - Use strict exact matching (no fuzzy matching)
    - Calculate accuracy as: 1 - (different_fields / total_fields)

    Args:
        predicted: Predicted JSON value (can be dict, list, or primitive)
        ground_truth: Ground truth JSON value
        path: Current path in JSON structure (for error reporting)

    Returns:
        Dict with:
        - total_fields: Total number of fields counted recursively
        - different_fields: Number of fields that differ
        - accuracy: 1 - (different_fields / total_fields)
        - differences: List of difference details
        - missing_fields: Fields in ground truth but not in predicted
        - extra_fields: Fields in predicted but not in ground truth
    """
    result = {
        'total_fields': 0,
        'different_fields': 0,
        'accuracy': 0.0,
        'differences': [],
        'missing_fields': [],
        'extra_fields': []
    }

    _compare_recursive(predicted, ground_truth, path, result)

    # Calculate accuracy using getomni formula
    if result['total_fields'] > 0:
        result['accuracy'] = 1.0 - (result['different_fields'] / result['total_fields'])
    else:
        result['accuracy'] = 1.0  # Empty comparison = perfect match

    return result


def _compare_recursive(
    predicted: Any,
    ground_truth: Any,
    path: str,
    result: Dict[str, Any]
) -> None:
    """
    Recursive helper for comparing JSON structures.

    Updates result dict in-place with field counts and differences.
    """
    # Type mismatch - count as different
    if type(predicted) != type(ground_truth):
        result['total_fields'] += 1
        result['different_fields'] += 1
        result['differences'].append({
            'path': path or 'root',
            'predicted': _safe_repr(predicted),
            'ground_truth': _safe_repr(ground_truth),
            'type': 'type_mismatch'
        })
        return

    # Handle None/null
    if predicted is None and ground_truth is None:
        result['total_fields'] += 1
        return

    # Handle dictionaries (objects)
    if isinstance(ground_truth, dict):
        _compare_dicts(predicted, ground_truth, path, result)
        return

    # Handle lists (arrays)
    if isinstance(ground_truth, list):
        _compare_lists(predicted, ground_truth, path, result)
        return

    # Handle primitives (string, number, boolean)
    result['total_fields'] += 1
    if predicted != ground_truth:
        result['different_fields'] += 1
        result['differences'].append({
            'path': path or 'root',
            'predicted': _safe_repr(predicted),
            'ground_truth': _safe_repr(ground_truth),
            'type': 'value_mismatch'
        })


def _compare_dicts(
    predicted: dict,
    ground_truth: dict,
    path: str,
    result: Dict[str, Any]
) -> None:
    """Compare two dictionaries recursively."""
    all_keys = set(predicted.keys()) | set(ground_truth.keys())

    for key in all_keys:
        field_path = f"{path}.{key}" if path else key

        # Missing field in predicted
        if key not in predicted:
            # Count all fields in ground truth value as different
            count = _count_fields(ground_truth[key])
            result['total_fields'] += count
            result['different_fields'] += count
            result['missing_fields'].append(field_path)
            result['differences'].append({
                'path': field_path,
                'predicted': '<missing>',
                'ground_truth': _safe_repr(ground_truth[key]),
                'type': 'missing_field'
            })
            continue

        # Extra field in predicted
        if key not in ground_truth:
            # Count all fields in predicted value as different
            count = _count_fields(predicted[key])
            result['total_fields'] += count
            result['different_fields'] += count
            result['extra_fields'].append(field_path)
            result['differences'].append({
                'path': field_path,
                'predicted': _safe_repr(predicted[key]),
                'ground_truth': '<missing>',
                'type': 'extra_field'
            })
            continue

        # Both have the field - recurse
        _compare_recursive(predicted[key], ground_truth[key], field_path, result)


def _compare_lists(
    predicted: list,
    ground_truth: list,
    path: str,
    result: Dict[str, Any]
) -> None:
    """Compare two lists item-by-item at same index."""
    max_len = max(len(predicted), len(ground_truth))

    for i in range(max_len):
        field_path = f"{path}[{i}]"

        # Item missing in predicted
        if i >= len(predicted):
            count = _count_fields(ground_truth[i])
            result['total_fields'] += count
            result['different_fields'] += count
            result['missing_fields'].append(field_path)
            result['differences'].append({
                'path': field_path,
                'predicted': '<missing>',
                'ground_truth': _safe_repr(ground_truth[i]),
                'type': 'missing_item'
            })
            continue

        # Extra item in predicted
        if i >= len(ground_truth):
            count = _count_fields(predicted[i])
            result['total_fields'] += count
            result['different_fields'] += count
            result['extra_fields'].append(field_path)
            result['differences'].append({
                'path': field_path,
                'predicted': _safe_repr(predicted[i]),
                'ground_truth': '<missing>',
                'type': 'extra_item'
            })
            continue

        # Both have the item - recurse
        _compare_recursive(predicted[i], ground_truth[i], field_path, result)


def _count_fields(value: Any) -> int:
    """Count total number of fields in a JSON value recursively."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return 1
    elif isinstance(value, dict):
        return sum(_count_fields(v) for v in value.values())
    elif isinstance(value, list):
        return sum(_count_fields(item) for item in value)
    else:
        return 1


def _safe_repr(value: Any, max_length: int = 100) -> str:
    """Safe string representation of a value, truncated if too long."""
    if value is None:
        return 'null'

    s = str(value)
    if len(s) > max_length:
        return s[:max_length] + '...'
    return s


def calculate_accuracy(comparison_result: Dict[str, Any]) -> float:
    """
    Extract accuracy percentage from comparison result.

    Args:
        comparison_result: Result from compare_json()

    Returns:
        Accuracy as float (0.0 to 1.0)
    """
    return comparison_result.get('accuracy', 0.0)


def format_diff_report(comparison_result: Dict[str, Any], max_diffs: int = 10) -> str:
    """
    Format comparison result as human-readable diff report.

    Args:
        comparison_result: Result from compare_json()
        max_diffs: Maximum number of differences to show

    Returns:
        Formatted string report
    """
    lines = []

    # Summary
    total = comparison_result['total_fields']
    diff = comparison_result['different_fields']
    acc = comparison_result['accuracy'] * 100

    lines.append(f"Total fields: {total}")
    lines.append(f"Different fields: {diff}")
    lines.append(f"Accuracy: {acc:.1f}%")
    lines.append("")

    # Missing/Extra fields summary
    if comparison_result['missing_fields']:
        lines.append(f"Missing fields: {len(comparison_result['missing_fields'])}")
    if comparison_result['extra_fields']:
        lines.append(f"Extra fields: {len(comparison_result['extra_fields'])}")

    if comparison_result['missing_fields'] or comparison_result['extra_fields']:
        lines.append("")

    # Show top differences
    differences = comparison_result['differences']
    if differences:
        lines.append(f"Top {min(max_diffs, len(differences))} differences:")
        for i, diff in enumerate(differences[:max_diffs], 1):
            lines.append(f"  {i}. {diff['path']}")
            lines.append(f"     Predicted: {diff['predicted']}")
            lines.append(f"     Ground Truth: {diff['ground_truth']}")
            lines.append(f"     Type: {diff['type']}")

    return '\n'.join(lines)
