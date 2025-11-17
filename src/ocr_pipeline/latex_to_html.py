"""
Simple LaTeX to HTML converter for tabular tables.

This module converts LaTeX \\begin{tabular}...\\end{tabular} tables to
semantic HTML <table> elements. It's designed specifically for the simple
tabular format produced by QwenVL.
"""

import re
from typing import List, Tuple


def parse_tabular_format(format_spec: str) -> List[str]:
    """
    Parse LaTeX tabular format specification.

    Example: {|l|c|r|} → ['left', 'center', 'right']

    Args:
        format_spec: LaTeX format string (e.g., "{|l|c|r|}")

    Returns:
        List of alignment values ('left', 'center', 'right')
    """
    # Remove braces and pipes
    clean_spec = format_spec.strip('{}|').replace('|', '')

    alignments = []
    for char in clean_spec:
        if char == 'l':
            alignments.append('left')
        elif char == 'c':
            alignments.append('center')
        elif char == 'r':
            alignments.append('right')
        else:
            # Default to left for unknown
            alignments.append('left')

    return alignments


def parse_tabular_row(row: str) -> List[str]:
    """
    Parse a LaTeX tabular row into cells.

    Example: "A & B & C" → ['A', 'B', 'C']

    Args:
        row: LaTeX row string (without trailing \\)

    Returns:
        List of cell contents (stripped)
    """
    # Split by & and strip whitespace
    cells = [cell.strip() for cell in row.split('&')]
    return cells


def latex_tabular_to_html(latex_table: str) -> str:
    """
    Convert LaTeX tabular table to HTML.

    Handles:
    - \\begin{tabular}{format} ... \\end{tabular}
    - Row separators: \\\\
    - Horizontal rules: \\hline
    - Column alignments: l (left), c (center), r (right)

    Args:
        latex_table: LaTeX table string

    Returns:
        HTML table string

    Example:
        Input:
            \\begin{tabular}{|l|c|r|}
            \\hline
            Name & Age & Score \\\\
            \\hline
            Alice & 25 & 95 \\\\
            Bob & 30 & 87 \\\\
            \\hline
            \\end{tabular}

        Output:
            <table>
            <thead>
            <tr><th align="left">Name</th><th align="center">Age</th><th align="right">Score</th></tr>
            </thead>
            <tbody>
            <tr><td align="left">Alice</td><td align="center">25</td><td align="right">95</td></tr>
            <tr><td align="left">Bob</td><td align="center">30</td><td align="right">87</td></tr>
            </tbody>
            </table>
    """
    # Extract format specification
    format_match = re.search(r'\\begin\{tabular\}\{([^}]+)\}', latex_table)
    if not format_match:
        # Fallback: no alignment info
        alignments = []
    else:
        format_spec = format_match.group(1)
        alignments = parse_tabular_format(format_spec)

    # Extract table content (between \begin{tabular} and \end{tabular})
    content_match = re.search(
        r'\\begin\{tabular\}\{[^}]+\}(.*?)\\end\{tabular\}',
        latex_table,
        re.DOTALL
    )

    if not content_match:
        return "<table></table>"  # Empty table

    content = content_match.group(1).strip()

    # Split into rows (by \\)
    raw_rows = content.split('\\\\')

    # Parse rows, filtering out \hline and empty rows
    rows = []
    for raw_row in raw_rows:
        row = raw_row.strip()
        # Skip \hline and empty rows
        if row and not row.startswith('\\hline'):
            cells = parse_tabular_row(row)
            rows.append(cells)

    if not rows:
        return "<table></table>"  # No rows

    # Determine header row (first row is typically header if followed by \hline)
    # For simplicity, treat first row as header
    has_header = len(rows) > 1  # Assume header if more than 1 row

    # Build HTML
    html_lines = ['<table>']

    if has_header:
        header_row = rows[0]
        html_lines.append('<thead>')
        html_lines.append('<tr>')
        for i, cell in enumerate(header_row):
            align = alignments[i] if i < len(alignments) else 'left'
            html_lines.append(f'<th align="{align}">{cell}</th>')
        html_lines.append('</tr>')
        html_lines.append('</thead>')

        data_rows = rows[1:]
    else:
        data_rows = rows

    # Data rows
    if data_rows:
        html_lines.append('<tbody>')
        for row in data_rows:
            html_lines.append('<tr>')
            for i, cell in enumerate(row):
                align = alignments[i] if i < len(alignments) else 'left'
                html_lines.append(f'<td align="{align}">{cell}</td>')
            html_lines.append('</tr>')
        html_lines.append('</tbody>')

    html_lines.append('</table>')

    return '\n'.join(html_lines)


def convert_markdown_latex_to_html(markdown: str) -> str:
    """
    Convert all LaTeX tables in markdown to HTML tables.

    Finds all \\begin{tabular}...\\end{tabular} blocks and replaces them
    with semantic HTML <table> elements.

    Args:
        markdown: Markdown content with LaTeX tables

    Returns:
        Markdown with HTML tables (LaTeX tables replaced)
    """
    # Find all tabular tables
    pattern = r'\\begin\{tabular\}.*?\\end\{tabular\}'

    def replace_table(match):
        latex_table = match.group(0)
        html_table = latex_tabular_to_html(latex_table)
        return html_table

    # Replace all LaTeX tables with HTML
    result = re.sub(pattern, replace_table, markdown, flags=re.DOTALL)

    return result
