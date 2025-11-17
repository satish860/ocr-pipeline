"""
Table Conversion Component.

Converts LaTeX tables in markdown to semantic HTML tables.
"""

import re
from typing import Dict, List

# Import LaTeX to HTML converter (conditional)
try:
    from .latex_to_html import convert_markdown_latex_to_html
except ImportError:
    # Fallback if module not available
    convert_markdown_latex_to_html = None


class TableConverter:
    """
    LaTeX to HTML table conversion component.

    Converts LaTeX tables in markdown to semantic HTML tables.
    """

    def convert(self, markdown: str, extracted_images: List[Dict]) -> Dict:
        """
        Convert LaTeX tables to HTML.

        Args:
            markdown: Markdown string with LaTeX tables
            extracted_images: List of extracted images (includes table images with base64 data)

        Returns:
            Dict with:
            - success: Boolean indicating success
            - markdown: Markdown with HTML tables (if conversion succeeded)
            - converted_count: Number of tables converted
            - error: Error message if success=False
        """
        try:
            # Check if LaTeX to HTML converter is available
            if convert_markdown_latex_to_html is None:
                # Converter not available, return original markdown
                return {
                    'success': True,
                    'markdown': markdown,
                    'converted_count': 0,
                    'error': 'LaTeX to HTML converter not available'
                }

            # Convert all LaTeX tables to HTML
            converted_markdown = convert_markdown_latex_to_html(markdown)

            # Count conversions (approximate by counting table tags)
            original_latex_count = len(re.findall(r'\\begin\{tabular\}', markdown))
            converted_html_count = len(re.findall(r'<table>', converted_markdown))

            return {
                'success': True,
                'markdown': converted_markdown,
                'converted_count': converted_html_count,
                'error': None
            }

        except Exception as e:
            # If conversion fails, return original markdown
            return {
                'success': False,
                'markdown': markdown,
                'converted_count': 0,
                'error': str(e)
            }
