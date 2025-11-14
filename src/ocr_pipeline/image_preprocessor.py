"""Image preprocessing utilities for OCR pipeline"""

import numpy as np
from PIL import Image
from typing import Union, Tuple
from pathlib import Path
import cv2


class ImagePreprocessor:
    """
    Preprocesses images before layout detection and OCR.

    Handles:
    - Deskewing (rotation correction)
    - Noise reduction
    - Contrast enhancement
    """

    def __init__(self):
        """Initialize ImagePreprocessor."""
        pass

    def deskew_image(self, image: Union[str, Path, Image.Image]) -> Tuple[Image.Image, float]:
        """
        Detect and correct skew/rotation in scanned documents.

        Args:
            image: PIL Image object or path to image file

        Returns:
            Tuple of (deskewed PIL Image, rotation angle in degrees)
        """
        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("image must be a file path or PIL Image object")

        # Convert PIL to OpenCV format
        image_np = np.array(pil_image)

        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            print("[INFO] No lines detected for deskewing, returning original image")
            return pil_image, 0.0

        # Calculate angles of all detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

        # Filter out outliers (keep angles close to horizontal/vertical)
        angles = [a for a in angles if abs(a) < 45 or abs(a - 90) < 45 or abs(a + 90) < 45]

        if len(angles) == 0:
            print("[INFO] No valid angles detected, returning original image")
            return pil_image, 0.0

        # Calculate median angle (more robust than mean)
        median_angle = np.median(angles)

        # Normalize angle to [-45, 45] range
        if median_angle > 45:
            median_angle -= 90
        elif median_angle < -45:
            median_angle += 90

        print(f"[INFO] Detected skew angle: {median_angle:.2f} degrees")

        # Only rotate if angle is significant (> 0.5 degrees)
        if abs(median_angle) < 0.5:
            print("[INFO] Skew angle too small, no rotation needed")
            return pil_image, 0.0

        # Rotate image to correct skew
        rotated = pil_image.rotate(
            -median_angle,  # Negative to correct the skew
            expand=True,    # Expand canvas to fit rotated image
            fillcolor='white'  # Fill gaps with white
        )

        print(f"[INFO] Rotated image by {-median_angle:.2f} degrees")

        return rotated, median_angle

    def enhance_contrast_clahe(self, image: Image.Image, clip_limit: float = 2.0, tile_size: int = 8) -> Image.Image:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Better than simple contrast enhancement for documents with varying lighting.

        Args:
            image: PIL Image
            clip_limit: Contrast limit (higher = more contrast, default 2.0)
            tile_size: Size of grid for histogram equalization (default 8)

        Returns:
            Contrast-enhanced PIL Image
        """
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert to LAB color space
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)

        # Split channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l_clahe = clahe.apply(l)

        # Merge channels
        lab_clahe = cv2.merge([l_clahe, a, b])

        # Convert back to BGR
        bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        # Convert back to PIL RGB
        rgb_clahe = cv2.cvtColor(bgr_clahe, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_clahe)

    def sharpen_image(self, image: Image.Image, amount: float = 1.5) -> Image.Image:
        """
        Apply sharpening to make text and edges clearer.

        Args:
            image: PIL Image
            amount: Sharpening amount (1.0 = no change, 2.0 = strong, default 1.5)

        Returns:
            Sharpened PIL Image
        """
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Sharpness(image)
        sharpened = enhancer.enhance(amount)
        return sharpened

    def denoise_image(self, image: Image.Image, strength: int = 10) -> Image.Image:
        """
        Apply denoising to reduce noise while preserving edges.

        Args:
            image: PIL Image
            strength: Denoising strength (default 10, higher = more denoising)

        Returns:
            Denoised PIL Image
        """
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Apply Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoisingColored(cv_image, None, strength, strength, 7, 21)

        # Convert back to PIL RGB
        rgb_denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_denoised)

    def upscale_image(self, image: Image.Image, scale: float = 1.5) -> Image.Image:
        """
        Upscale image for better detection of small text.

        Args:
            image: PIL Image
            scale: Upscale factor (1.5 = 50% larger, 2.0 = double size)

        Returns:
            Upscaled PIL Image
        """
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)

        # Use LANCZOS resampling for best quality
        upscaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return upscaled

    def preprocess_for_ocr(
        self,
        image: Union[str, Path, Image.Image],
        deskew: bool = True,
        enhance_contrast: bool = True,
        clahe: bool = False,
        sharpen: bool = False,
        denoise: bool = False,
        upscale: float = 1.0
    ) -> Image.Image:
        """
        Full preprocessing pipeline for OCR and layout detection.

        Args:
            image: PIL Image object or path to image file
            deskew: Whether to apply deskew correction
            enhance_contrast: Whether to enhance contrast (simple method)
            clahe: Use CLAHE instead of simple contrast (better, but slower)
            sharpen: Apply sharpening filter
            denoise: Apply denoising (slow, use for noisy images only)
            upscale: Upscale factor (1.0 = no upscaling, 1.5 = 50% larger)

        Returns:
            Preprocessed PIL Image
        """
        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("image must be a file path or PIL Image object")

        processed = pil_image

        # Step 1: Upscale (if requested, do first for better subsequent processing)
        if upscale > 1.0:
            processed = self.upscale_image(processed, upscale)

        # Step 2: Deskew
        if deskew:
            processed, angle = self.deskew_image(processed)

        # Step 3: Denoise (if requested, do before contrast enhancement)
        if denoise:
            processed = self.denoise_image(processed)

        # Step 4: Enhance contrast (CLAHE or simple)
        if clahe:
            processed = self.enhance_contrast_clahe(processed)
        elif enhance_contrast:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(1.2)  # Slightly increase contrast

        # Step 5: Sharpen (do last for best results)
        if sharpen:
            processed = self.sharpen_image(processed)

        return processed
