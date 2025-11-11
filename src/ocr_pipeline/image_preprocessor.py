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

    def preprocess_for_ocr(
        self,
        image: Union[str, Path, Image.Image],
        deskew: bool = True,
        enhance_contrast: bool = True
    ) -> Image.Image:
        """
        Full preprocessing pipeline for OCR.

        Args:
            image: PIL Image object or path to image file
            deskew: Whether to apply deskew correction
            enhance_contrast: Whether to enhance contrast

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

        # Step 1: Deskew
        if deskew:
            processed, angle = self.deskew_image(processed)

        # Step 2: Enhance contrast (optional)
        if enhance_contrast:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(1.2)  # Slightly increase contrast

        return processed
