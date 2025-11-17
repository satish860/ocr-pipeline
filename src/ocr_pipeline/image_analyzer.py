"""
Image Analysis and Preprocessing Component.

Handles image quality analysis and preprocessing to improve OCR accuracy.
"""

from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


class ImageAnalyzer:
    """
    Image quality analysis and preprocessing component.

    Analyzes image quality metrics (sharpness, contrast, resolution) and
    applies preprocessing to improve OCR accuracy.
    """

    def analyze(self, image: Image.Image) -> Dict:
        """
        Analyze image quality metrics to determine if preprocessing is needed.

        Args:
            image: PIL Image object

        Returns:
            Dict with:
            - success: Boolean indicating success
            - quality: Dict with quality metrics (sharpness, contrast, resolution, width, height, needs_preprocessing)
            - error: Error message if success=False
        """
        try:
            # Convert to grayscale for analysis
            gray = image.convert('L')
            gray_array = np.array(gray)

            # Sharpness: Laplacian variance
            # Higher values = sharper image
            # Typical ranges: <50 = blurry, 50-100 = moderate, >100 = sharp
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            filtered = np.abs(np.convolve(gray_array.flatten(), laplacian.flatten(), mode='same'))
            sharpness = filtered.var()

            # Contrast: Standard deviation of pixel values
            # Higher values = better contrast
            # Typical ranges: <30 = low contrast, 30-60 = moderate, >60 = high contrast
            contrast = gray_array.std()

            # Resolution
            width, height = image.size
            resolution = width * height

            quality_metrics = {
                'sharpness': float(sharpness),
                'contrast': float(contrast),
                'resolution': resolution,
                'width': width,
                'height': height,
                'needs_preprocessing': sharpness < 100 or contrast < 50 or resolution < 1000000
            }

            return {
                'success': True,
                'quality': quality_metrics,
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'quality': None,
                'error': str(e)
            }

    def preprocess(
        self,
        image: Image.Image,
        quality_metrics: Optional[Dict] = None,
        enhance_contrast: bool = True,
        sharpen: bool = True,
        upscale: bool = True,
        max_pixels: int = 4608 * 32 * 32
    ) -> Dict:
        """
        Preprocess image to improve OCR quality.

        Applies conservative enhancements that improve readability without
        over-processing or creating artifacts.

        Args:
            image: PIL Image object
            quality_metrics: Optional quality metrics from analyze() to determine if preprocessing is needed
            enhance_contrast: Apply CLAHE-style contrast enhancement
            sharpen: Apply unsharp mask sharpening
            upscale: Upscale if resolution is too low (respects max_pixels)
            max_pixels: Maximum total pixels (default: 4608*32*32 for QwenVL)

        Returns:
            Dict with:
            - success: Boolean indicating success
            - image: Preprocessed PIL Image
            - applied: Boolean indicating if preprocessing was applied
            - error: Error message if success=False
        """
        try:
            # Check if preprocessing is needed
            if quality_metrics and not quality_metrics.get('needs_preprocessing', True):
                print("Image quality is good, skipping preprocessing")
                return {
                    'success': True,
                    'image': image,
                    'applied': False,
                    'error': None
                }

            result = image.copy()
            preprocessing_applied = False

            # Step 1: Upscale if resolution is too low, but respect max_pixels
            if upscale:
                width, height = result.size
                current_pixels = width * height

                # Only upscale if current resolution is low AND won't exceed max_pixels
                # Target: use 70% of max_pixels to leave headroom for QwenVL processing
                target_pixels = int(max_pixels * 0.7)

                if current_pixels < target_pixels:
                    scale_factor = (target_pixels / current_pixels) ** 0.5
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)

                    # Double-check we don't exceed max_pixels
                    if new_width * new_height <= max_pixels:
                        # Use Lanczos resampling (high quality)
                        result = result.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        print(f"  Upscaled: {width}x{height} -> {new_width}x{new_height} "
                              f"({current_pixels} -> {new_width*new_height} pixels)")
                        preprocessing_applied = True
                    else:
                        print(f"  Skipped upscaling: would exceed max_pixels ({max_pixels})")

            # Step 2: Enhance contrast (CLAHE approximation using Pillow)
            if enhance_contrast:
                # Convert to LAB color space equivalent using RGB channels
                enhancer = ImageEnhance.Contrast(result)
                result = enhancer.enhance(1.3)  # 30% contrast boost
                print(f"  Applied contrast enhancement")
                preprocessing_applied = True

            # Step 3: Sharpen to enhance text edges
            if sharpen:
                # Unsharp mask: enhances edges without creating artifacts
                result = result.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
                print(f"  Applied sharpening")
                preprocessing_applied = True

            return {
                'success': True,
                'image': result,
                'applied': preprocessing_applied,
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'image': image,
                'applied': False,
                'error': str(e)
            }
