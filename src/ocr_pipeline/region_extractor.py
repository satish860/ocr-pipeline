"""Region extraction from images based on bounding boxes"""

from typing import Dict, List, Union
from pathlib import Path
from PIL import Image
import os


class RegionExtractor:
    """
    Extracts image regions based on bounding box coordinates.

    Works with layout detection results to crop and save individual regions.
    """

    def __init__(self):
        """Initialize RegionExtractor."""
        pass

    def crop_region(self, image: Image.Image, bbox: List[int]) -> Image.Image:
        """
        Crop a single region from an image using bounding box coordinates.

        Args:
            image: PIL Image object
            bbox: Bounding box coordinates [x1, y1, x2, y2]

        Returns:
            Cropped PIL Image
        """
        if len(bbox) != 4:
            raise ValueError(f"bbox must have 4 coordinates, got {len(bbox)}")

        x1, y1, x2, y2 = bbox

        # Ensure coordinates are within image bounds
        width, height = image.size
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        # Ensure x2 > x1 and y2 > y1
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox coordinates: {bbox}")

        # Crop the region
        cropped = image.crop((x1, y1, x2, y2))

        return cropped

    def extract_regions(
        self,
        image_input: Union[str, Path, Image.Image],
        layout_result: Dict
    ) -> List[Dict]:
        """
        Extract all regions from an image based on layout detection results.

        Args:
            image_input: Path to image file or PIL Image object
            layout_result: Result dictionary from LayoutDetector.detect_layout()

        Returns:
            List of dictionaries containing:
            - index: Region index (1-based)
            - type: Element type (table, paragraph, header, etc.)
            - bbox: Bounding box coordinates
            - image: Cropped PIL Image
            - width: Cropped region width
            - height: Cropped region height
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("image_input must be a file path or PIL Image object")

        # Extract regions
        regions = []
        elements = layout_result.get('elements', [])

        for idx, element in enumerate(elements, 1):
            bbox = element['bbox']
            element_type = element['type']

            try:
                # Crop the region
                cropped = self.crop_region(image, bbox)

                regions.append({
                    'index': idx,
                    'type': element_type,
                    'bbox': bbox,
                    'image': cropped,
                    'width': cropped.width,
                    'height': cropped.height,
                    'content_preview': element.get('content', '')[:100]
                })

            except ValueError as e:
                print(f"[WARNING] Failed to crop region {idx}: {e}")
                continue

        return regions

    def save_regions(
        self,
        regions: List[Dict],
        output_dir: Union[str, Path] = "output",
        save_images: bool = False
    ) -> List[str]:
        """
        Save extracted regions to files.

        Args:
            regions: List of region dictionaries from extract_regions()
            output_dir: Directory to save region images
            save_images: Whether to save individual region images (default: False)

        Returns:
            List of saved file paths (empty if save_images=False)
        """
        if not save_images:
            return []

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []

        for region in regions:
            idx = region['index']
            element_type = region['type']
            image = region['image']

            # Generate filename
            filename = f"region_{idx}_{element_type}.png"
            filepath = output_path / filename

            # Save image
            image.save(filepath)
            saved_files.append(str(filepath))

            print(f"[SAVED] {filepath} ({region['width']}x{region['height']})")

        return saved_files

    def extract_and_save(
        self,
        image_input: Union[str, Path, Image.Image],
        layout_result: Dict,
        output_dir: Union[str, Path] = "output",
        save_images: bool = False
    ) -> List[Dict]:
        """
        Convenience method to extract and save regions in one call.

        Args:
            image_input: Path to image file or PIL Image object
            layout_result: Result dictionary from LayoutDetector.detect_layout()
            output_dir: Directory to save region images
            save_images: Whether to save individual region images (default: False)

        Returns:
            List of region dictionaries with added 'filepath' key (if save_images=True)
        """
        # Extract regions
        regions = self.extract_regions(image_input, layout_result)

        # Save regions
        saved_files = self.save_regions(regions, output_dir, save_images)

        # Add filepath to each region
        for region, filepath in zip(regions, saved_files):
            region['filepath'] = filepath

        return regions
