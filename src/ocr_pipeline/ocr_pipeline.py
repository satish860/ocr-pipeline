"""
OCR Pipeline Orchestrator.

Main orchestrator that coordinates ImageAnalyzer, QwenExtractor, TableConverter,
and ClaudeRefiner components to perform end-to-end document extraction.
"""

from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

from PIL import Image

from .image_analyzer import ImageAnalyzer
from .qwen_extractor import QwenExtractor
from .table_converter import TableConverter

# Import Claude refiner (conditional to avoid import errors)
try:
    from .claude_refiner import refine_with_claude, refine_with_agentic_loop
except ImportError:
    # Fallback if module not available
    refine_with_claude = None
    refine_with_agentic_loop = None


class OCRPipeline:
    """
    Main OCR pipeline orchestrator.

    Coordinates the ImageAnalyzer, QwenExtractor, TableConverter, and ClaudeRefiner
    components to perform end-to-end document extraction.
    """

    def __init__(
        self,
        preprocess: bool = True,
        refine: bool = False,
        agentic_refine: bool = False,
        max_refinement_iterations: int = 2,
        convert_tables_to_html: bool = False,
        output_dir: Optional[str] = None,
        min_pixels: int = 512 * 32 * 32,
        max_pixels: int = 4608 * 32 * 32
    ):
        """
        Initialize OCR pipeline.

        Args:
            preprocess: Whether to apply image preprocessing (default: True)
            refine: Whether to refine extraction using Claude Sonnet 4.5 (default: False)
            agentic_refine: Whether to use iterative agentic refinement (default: False)
            max_refinement_iterations: Maximum iterations for agentic refinement (default: 2)
            convert_tables_to_html: Whether to convert LaTeX tables to HTML (default: False)
            output_dir: Directory to save outputs (QwenVL + each iteration). If None, nothing saved.
            min_pixels: Minimum pixels for image resize
            max_pixels: Maximum pixels for image resize
        """
        self.preprocess = preprocess
        self.refine = refine
        self.agentic_refine = agentic_refine
        self.max_refinement_iterations = max_refinement_iterations
        self.convert_tables_to_html = convert_tables_to_html
        self.output_dir = output_dir

        # Initialize components
        self.analyzer = ImageAnalyzer()
        self.extractor = QwenExtractor(min_pixels, max_pixels)
        self.table_converter = TableConverter() if convert_tables_to_html else None
        self.refiner = None  # Lazy initialization in apply()

    def apply(
        self,
        image_path: str,
        include_images: bool = True,
        include_usage: bool = False
    ) -> Dict:
        """
        Apply OCR pipeline to extract document content.

        Main entry point for document extraction. Returns markdown with optional
        inline base64 images and a separate array of extracted images.

        Args:
            image_path: Path to the image file
            include_images: Whether to extract and embed images
            include_usage: Whether to include usage/cost data in response

        Returns:
            Dict with:
            - success: Boolean indicating success
            - markdown: Markdown with inline base64 images (if include_images=True)
                        If refine=True, markdown will be HTML instead of LaTeX
            - images: List of extracted images [{type, base64, bbox}]
            - elements: List of detected elements with coordinates
            - usage: Usage/cost data (if include_usage=True)
            - quality: Image quality metrics (if preprocess=True)
            - refinement: Refinement metadata (if refine=True)
            - error: Error message if success=False
        """
        try:
            # Step 1: Load original image
            original_image = Image.open(image_path)
            image_to_process = original_image
            quality_result = None

            # Step 2: Analyze and preprocess (if enabled)
            if self.preprocess:
                # Analyze image quality
                quality_result = self.analyzer.analyze(original_image)

                if not quality_result['success']:
                    print(f"Warning: Quality analysis failed: {quality_result['error']}")
                else:
                    quality_metrics = quality_result['quality']
                    print(f"Image quality: sharpness={quality_metrics['sharpness']:.1f}, "
                          f"contrast={quality_metrics['contrast']:.1f}, "
                          f"resolution={quality_metrics['width']}x{quality_metrics['height']}")

                    # Apply preprocessing if needed
                    if quality_metrics['needs_preprocessing']:
                        print("Preprocessing image...")
                        preprocess_result = self.analyzer.preprocess(
                            original_image,
                            quality_metrics=quality_metrics
                        )

                        if preprocess_result['success']:
                            image_to_process = preprocess_result['image']
                            quality_metrics['preprocessing_applied'] = preprocess_result['applied']
                        else:
                            print(f"Warning: Preprocessing failed: {preprocess_result['error']}")
                            quality_metrics['preprocessing_applied'] = False
                    else:
                        print("Image quality is good, skipping preprocessing")
                        quality_metrics['preprocessing_applied'] = False

            # Step 3: Extract with QwenVL
            print("Extracting with QwenVL...")
            extraction_result = self.extractor.extract(
                image_to_process,
                include_images=include_images,
                include_usage=include_usage
            )

            if not extraction_result['success']:
                return {
                    **extraction_result,
                    'quality': quality_result['quality'] if quality_result else None
                }

            # Save QwenVL output if output_dir specified
            if self.output_dir:
                self._save_output(
                    content=extraction_result['markdown'],
                    image_path=image_path,
                    stage="0_qwen_original"
                )

            # Step 4: Convert tables to HTML (if enabled)
            if self.convert_tables_to_html and self.table_converter:
                print("Converting LaTeX tables to HTML...")
                conversion_result = self.table_converter.convert(
                    extraction_result['markdown'],
                    extraction_result['images']
                )

                if conversion_result['success']:
                    extraction_result['markdown'] = conversion_result['markdown']
                    print(f"Converted {conversion_result['converted_count']} tables to HTML")
                else:
                    print(f"Warning: Table conversion failed: {conversion_result['error']}")

            # Step 5: Refine with Claude (if enabled)
            if self.agentic_refine:
                # Use iterative agentic refinement
                if refine_with_agentic_loop is None:
                    print("WARNING: Claude refiner not available, skipping refinement")
                else:
                    print("Starting agentic refinement...")
                    extraction_result = refine_with_agentic_loop(
                        image_to_process,
                        extraction_result,
                        max_iterations=self.max_refinement_iterations,
                        include_usage=include_usage
                    )

                    # Save iteration outputs if output_dir specified
                    if self.output_dir and 'refinement' in extraction_result:
                        refinement = extraction_result['refinement']
                        if refinement.get('iteration_history'):
                            for iter_data in refinement['iteration_history']:
                                iteration_num = iter_data['iteration']
                                self._save_output(
                                    content=iter_data['html'],
                                    image_path=image_path,
                                    stage=f"{iteration_num}_iteration_{iteration_num}"
                                )

            elif self.refine:
                # Use single-pass refinement
                if refine_with_claude is None:
                    print("WARNING: Claude refiner not available, skipping refinement")
                else:
                    print("Refining with Claude Sonnet 4.5...")
                    extraction_result = refine_with_claude(
                        image_to_process,
                        extraction_result,
                        include_usage=include_usage
                    )

                    # Save single-pass refinement output if output_dir specified
                    if self.output_dir:
                        self._save_output(
                            content=extraction_result['markdown'],
                            image_path=image_path,
                            stage="1_refined"
                        )

            # Step 6: Prepare final result
            final_result = {
                **extraction_result,
                'quality': quality_result['quality'] if quality_result else None
            }

            return final_result

        except Exception as e:
            return {
                'success': False,
                'markdown': '',
                'images': [],
                'elements': [],
                'usage': {},
                'quality': None,
                'error': str(e)
            }

    def _save_output(self, content: str, image_path: str, stage: str):
        """
        Save output to file in output directory.

        Args:
            content: Markdown/HTML content to save
            image_path: Original image path (used for filename)
            stage: Stage name (e.g., "0_qwen_original", "1_iteration_1", "2_iteration_2")
        """
        if not self.output_dir:
            return

        # Create output directory if it doesn't exist
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename: image_name_stage_timestamp.md
        image_name = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_name}_{stage}_{timestamp}.md"

        output_file = output_path / filename

        # Save content
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Saved: {output_file}")
