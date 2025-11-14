"""Centralized preprocessing pipeline for OCR"""

from typing import Dict, Tuple
from PIL import Image
from .image_preprocessor import ImagePreprocessor


class PreprocessingPipeline:
    """
    Centralized preprocessing pipeline for OCR.

    Single method that always performs the same steps:
    1. Coarse rotation correction (90/180/270 degrees)
    2. Fine deskew (small angle corrections)
    3. Enhancement (CLAHE + sharpening for better layout detection)
    """

    def __init__(self):
        """Initialize preprocessing pipeline."""
        self.preprocessor = ImagePreprocessor()

    def preprocess(
        self,
        image: Image.Image,
        rotation_degrees: int = 0
    ) -> Tuple[Image.Image, Dict]:
        """
        Preprocess image for OCR pipeline.

        Args:
            image: Input PIL Image
            rotation_degrees: Coarse rotation angle from classification (0, 90, 180, 270)

        Returns:
            Tuple of (preprocessed_image, metadata)
            metadata contains:
                - rotation_applied: Coarse rotation applied
                - deskew_angle: Fine angle correction applied
        """
        metadata = {
            'rotation_applied': rotation_degrees,
            'deskew_angle': 0.0
        }

        # Step 1: Apply coarse rotation (from document classification)
        if rotation_degrees != 0:
            # PIL rotation is counter-clockwise, so negate
            image = image.rotate(-rotation_degrees, expand=True)

        # Step 2: Fine deskew (small angle corrections)
        deskewed_image, deskew_angle = self.preprocessor.deskew_image(image)
        metadata['deskew_angle'] = deskew_angle
        image = deskewed_image

        # Step 3: Enhancement (always applied for best layout detection)
        image = self.preprocessor.preprocess_for_ocr(
            image,
            deskew=False,  # Already deskewed above
            enhance_contrast=False,  # Use CLAHE instead
            clahe=True,  # Better contrast for layout detection
            sharpen=True,  # Sharpen to make lines/borders clearer
            denoise=False,  # Skip (slow and not needed)
            upscale=1.0  # No upscaling needed
        )

        return image, metadata
