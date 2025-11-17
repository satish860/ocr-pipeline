"""
Refinement Analyzer.

Analyzes extracted content to determine if refinement is needed.
Checks for presence of tables, images, infographics, and charts.
"""

import re
from typing import Dict


class RefinementAnalyzer:
    """
    Analyzes extracted content to determine if refinement is needed.
    
    Refinement is required when the page contains:
    - Tables (LaTeX, HTML, or Markdown)
    - Embedded images
    """

    def __init__(self):
        """Initialize the refinement analyzer."""
        pass

    def needs_refinement(self, markdown: str, images: list) -> Dict:
        """
        Analyze if content needs refinement.

        Refinement is needed if tables or images are present.

        Args:
            markdown: Extracted markdown content
            images: List of extracted images

        Returns:
            Dict with:
            - needs_refinement: Boolean indicating if refinement is needed
            - reasons: List of reasons why refinement is needed
            - content_types: Dict with counts of detected content types
            - confidence: Float between 0 and 1 indicating confidence
        """
        reasons = []
        content_types = {
            'tables': 0,
            'images': 0
        }
        confidence = 0.0

        # Check for tables
        table_count = self._detect_tables(markdown)
        if table_count > 0:
            content_types['tables'] = table_count
            reasons.append(f"Found {table_count} table(s)")
            confidence = 1.0

        # Check for images
        image_count = self._detect_images(markdown, images)
        if image_count > 0:
            content_types['images'] = image_count
            reasons.append(f"Found {image_count} image(s)")
            confidence = 1.0

        needs_refinement = len(reasons) > 0

        return {
            'needs_refinement': needs_refinement,
            'reasons': reasons,
            'content_types': content_types,
            'confidence': confidence
        }

    def _detect_tables(self, markdown: str) -> int:
        """
        Detect tables in markdown content.

        Looks for:
        - LaTeX table environments (\\begin{table}, \\begin{tabular})
        - HTML table tags (<table>)
        - Markdown table syntax (|---|)

        Args:
            markdown: Markdown content

        Returns:
            Count of detected tables
        """
        count = 0

        # LaTeX tables
        latex_table_pattern = r'\\begin\{(?:table|tabular|array)\}'
        count += len(re.findall(latex_table_pattern, markdown, re.IGNORECASE))

        # HTML tables
        html_table_pattern = r'<table[^>]*>'
        count += len(re.findall(html_table_pattern, markdown, re.IGNORECASE))

        # Markdown tables (lines with | separators)
        markdown_table_pattern = r'\|[^\|]+\|[^\|]*\n\|[\s\-\|:]+\|'
        count += len(re.findall(markdown_table_pattern, markdown))

        return count

    def _detect_images(self, markdown: str, images: list) -> int:
        """
        Detect images in content.

        Looks for:
        - Markdown image syntax ![alt](url)
        - HTML img tags
        - Extracted images in the images list

        Args:
            markdown: Markdown content
            images: List of extracted images

        Returns:
            Count of detected images
        """
        count = 0

        # Markdown images
        markdown_img_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
        count += len(re.findall(markdown_img_pattern, markdown))

        # HTML images
        html_img_pattern = r'<img[^>]*>'
        count += len(re.findall(html_img_pattern, markdown, re.IGNORECASE))

        # Extracted images
        if images and isinstance(images, list):
            count += len(images)

        return count

