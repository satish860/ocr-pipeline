"""Spatial relationship analysis for document regions"""

from typing import List, Dict, Tuple


class SpatialAnalyzer:
    """
    Analyzes spatial relationships between document regions.

    Uses bounding box coordinates to detect:
    - Vertical alignment (above/below)
    - Horizontal alignment (left/right)
    - Containment (inside)
    - Proximity (nearby)
    """

    def __init__(self, proximity_threshold: int = 100):
        """
        Initialize SpatialAnalyzer.

        Args:
            proximity_threshold: Distance in pixels to consider regions "nearby"
        """
        self.proximity_threshold = proximity_threshold

    def analyze_relationships(self, regions: List[Dict]) -> List[Dict]:
        """
        Analyze spatial relationships between all regions.

        Args:
            regions: List of region dictionaries with 'bbox', 'type', 'index', etc.

        Returns:
            List of regions with added 'relationships' field
        """
        enriched_regions = []

        for i, region in enumerate(regions):
            relationships = {
                'above': [],
                'below': [],
                'left': [],
                'right': [],
                'inside': [],
                'nearby': []
            }

            for j, other_region in enumerate(regions):
                if i == j:
                    continue

                rel_type = self._detect_relationship(region['bbox'], other_region['bbox'])

                if rel_type:
                    relationships[rel_type].append({
                        'index': other_region['index'],
                        'type': other_region['type']
                    })

            # Add relationships to region
            region_copy = region.copy()
            region_copy['relationships'] = relationships
            enriched_regions.append(region_copy)

        return enriched_regions

    def _detect_relationship(self, bbox1: List[int], bbox2: List[int]) -> str:
        """
        Detect spatial relationship between two bounding boxes.

        Args:
            bbox1: [x1, y1, x2, y2] of first region
            bbox2: [x1, y1, x2, y2] of second region

        Returns:
            Relationship type: 'above', 'below', 'left', 'right', 'inside', 'nearby', or None
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Check containment (bbox2 inside bbox1)
        if (x1_1 <= x1_2 and x2_2 <= x2_1 and
            y1_1 <= y1_2 and y2_2 <= y2_1):
            return 'inside'

        # Calculate centers
        center_x1 = (x1_1 + x2_1) / 2
        center_y1 = (y1_1 + y2_1) / 2
        center_x2 = (x1_2 + x2_2) / 2
        center_y2 = (y1_2 + y2_2) / 2

        # Check horizontal overlap (same vertical strip)
        h_overlap = self._check_overlap(x1_1, x2_1, x1_2, x2_2)

        # Check vertical overlap (same horizontal strip)
        v_overlap = self._check_overlap(y1_1, y2_1, y1_2, y2_2)

        # Vertical relationships (above/below) - require horizontal overlap
        if h_overlap > 0.3:  # At least 30% horizontal overlap
            if center_y2 < center_y1:  # bbox2 is above bbox1
                return 'above'
            elif center_y2 > center_y1:  # bbox2 is below bbox1
                return 'below'

        # Horizontal relationships (left/right) - require vertical overlap
        if v_overlap > 0.3:  # At least 30% vertical overlap
            if center_x2 < center_x1:  # bbox2 is to the left of bbox1
                return 'left'
            elif center_x2 > center_x1:  # bbox2 is to the right of bbox1
                return 'right'

        # Check proximity (close but not aligned)
        distance = ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5
        if distance < self.proximity_threshold:
            return 'nearby'

        return None

    def _check_overlap(self, start1: int, end1: int, start2: int, end2: int) -> float:
        """
        Calculate overlap ratio between two 1D ranges.

        Args:
            start1, end1: First range
            start2, end2: Second range

        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_start >= overlap_end:
            return 0.0

        overlap = overlap_end - overlap_start
        range1_len = end1 - start1
        range2_len = end2 - start2

        # Return ratio relative to smaller range
        min_len = min(range1_len, range2_len)
        return overlap / min_len if min_len > 0 else 0.0

    def group_by_vertical_alignment(self, regions: List[Dict]) -> List[List[Dict]]:
        """
        Group regions that are vertically aligned (same Y-range).
        Useful for identifying rows in tables or horizontally-aligned content.

        Args:
            regions: List of regions with relationships

        Returns:
            List of groups, where each group contains vertically-aligned regions
        """
        groups = []
        processed = set()

        for i, region in enumerate(regions):
            if i in processed:
                continue

            group = [region]
            processed.add(i)

            # Find all regions with significant vertical overlap
            for j, other_region in enumerate(regions):
                if j in processed or i == j:
                    continue

                v_overlap = self._check_overlap(
                    region['bbox'][1], region['bbox'][3],
                    other_region['bbox'][1], other_region['bbox'][3]
                )

                if v_overlap > 0.5:  # 50% vertical overlap
                    group.append(other_region)
                    processed.add(j)

            # Sort group by X-coordinate (left to right)
            group.sort(key=lambda r: r['bbox'][0])
            groups.append(group)

        # Sort groups by Y-coordinate (top to bottom)
        groups.sort(key=lambda g: g[0]['bbox'][1])

        return groups

    def merge_related_regions(self, regions: List[Dict]) -> List[Dict]:
        """
        Merge regions that are spatially related.

        For example:
        - Table + right-aligned handwritten text → Combined table with extra column
        - Paragraph + left margin note → Paragraph with note

        Args:
            regions: List of regions with relationships

        Returns:
            List of merged regions with spatial context preserved
        """
        merged_regions = []
        processed = set()

        for i, region in enumerate(regions):
            if i in processed:
                continue

            # Check if this region has related regions to the right (common for annotations)
            right_regions = region.get('relationships', {}).get('right', [])

            if right_regions:
                # Collect all right-aligned content
                related_content = []
                for rel in right_regions:
                    rel_idx = next((idx for idx, r in enumerate(regions) if r['index'] == rel['index']), None)
                    if rel_idx is not None:
                        related_content.append(regions[rel_idx])
                        processed.add(rel_idx)

                # Create merged region
                merged_region = region.copy()
                merged_region['related_right'] = related_content
                merged_regions.append(merged_region)
            else:
                merged_regions.append(region)

            processed.add(i)

        return merged_regions
