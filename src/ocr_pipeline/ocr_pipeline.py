"""
OCR Pipeline Orchestrator.

Main orchestrator that coordinates ImageAnalyzer, QwenExtractor, TableConverter,
and ClaudeRefiner components to perform end-to-end document extraction.
"""

from typing import Dict, Optional
from pathlib import Path
from datetime import datetime
import time

from PIL import Image

from .image_analyzer import ImageAnalyzer
from .qwen_extractor import QwenExtractor
from .table_converter import TableConverter
from .refinement_analyzer import RefinementAnalyzer
from .retry_utils import retry_with_backoff

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
        auto_refine: bool = False,
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
            auto_refine: Whether to automatically refine only if content needs it (default: False)
            max_refinement_iterations: Maximum iterations for agentic refinement (default: 2)
            convert_tables_to_html: Whether to convert LaTeX tables to HTML (default: False)
            output_dir: Directory to save outputs (QwenVL + each iteration). If None, nothing saved.
            min_pixels: Minimum pixels for image resize
            max_pixels: Maximum pixels for image resize
        """
        self.preprocess = preprocess
        self.refine = refine
        self.agentic_refine = agentic_refine
        self.auto_refine = auto_refine
        self.max_refinement_iterations = max_refinement_iterations
        self.convert_tables_to_html = convert_tables_to_html
        self.output_dir = output_dir

        # Initialize components
        self.analyzer = ImageAnalyzer()
        self.extractor = QwenExtractor(min_pixels, max_pixels)
        self.table_converter = TableConverter() if convert_tables_to_html else None
        self.refinement_analyzer = RefinementAnalyzer()
        self.refiner = None  # Lazy initialization in apply()

    def apply(
        self,
        image_path,
        include_images: bool = True,
        include_usage: bool = False
    ) -> Dict:
        """
        Apply OCR pipeline to extract document content.

        Main entry point for document extraction. Returns markdown with optional
        inline base64 images and a separate array of extracted images.

        Args:
            image_path: Path to the image file or PIL Image object
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
            - refinement_analysis: Analysis of whether refinement is needed (if auto_refine=True)
            - error: Error message if success=False
        """
        try:
            # Step 1: Load original image
            if isinstance(image_path, Image.Image):
                original_image = image_path
            else:
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

            # Step 3: Extract with QwenVL (with retry logic)
            print("Extracting with QwenVL...")
            extraction_result = self._extract_with_retry(
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

            # Step 5: Analyze if refinement is needed
            refinement_analysis = None
            should_refine = False
            
            if self.auto_refine:
                # Auto-refine mode: Only refine if content analysis detects complex elements
                refinement_analysis = self.refinement_analyzer.needs_refinement(
                    extraction_result['markdown'],
                    extraction_result.get('images', [])
                )
                should_refine = refinement_analysis['needs_refinement']
                
                if should_refine:
                    print(f"Refinement needed (confidence: {refinement_analysis['confidence']:.1%}):")
                    for reason in refinement_analysis['reasons']:
                        print(f"  - {reason}")
                else:
                    print("No refinement needed - content is simple text only")
            else:
                # Manual mode: Refine if user explicitly enabled it
                should_refine = self.refine or self.agentic_refine

            # Step 6: Refine with Claude (if enabled or auto-detected)
            if should_refine and self.agentic_refine:
                # Use iterative agentic refinement
                if refine_with_agentic_loop is None:
                    print("WARNING: Claude refiner not available, skipping refinement")
                else:
                    print("Starting agentic refinement...")
                    # Note: We pass original_image (not preprocessed) to Claude
                    # Preprocessing is only for QwenVL extraction, Claude works fine with original
                    extraction_result = refine_with_agentic_loop(
                        original_image,
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

            elif should_refine and self.refine:
                # Use single-pass refinement
                if refine_with_claude is None:
                    print("WARNING: Claude refiner not available, skipping refinement")
                else:
                    print("Refining with Claude Sonnet 4.5...")
                    # Note: We pass original_image (not preprocessed) to Claude
                    # Preprocessing is only for QwenVL extraction, Claude works fine with original
                    extraction_result = refine_with_claude(
                        original_image,
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

            # Step 7: Prepare final result
            final_result = {
                **extraction_result,
                'quality': quality_result['quality'] if quality_result else None,
                'refinement_analysis': refinement_analysis
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

    async def apply_fast(
        self,
        image_path,
        include_images: bool = True,
        include_usage: bool = False
    ) -> Dict:
        """
        Apply OCR pipeline to extract document content (async).

        Simplified version with only QwenVL extraction and LaTeX table conversion.

        Args:
            image_path: Path to the image file or PIL Image object
            include_images: Whether to extract and embed images
            include_usage: Whether to include usage/cost data in response

        Returns:
            Dict with:
            - success: Boolean indicating success
            - markdown: Markdown with inline base64 images (if include_images=True)
            - images: List of extracted images [{type, base64, bbox}]
            - elements: List of detected elements with coordinates
            - usage: Usage/cost data (if include_usage=True)
            - error: Error message if success=False
        """
        try:

            # Step 1: Load image
            if isinstance(image_path, Image.Image):
                image_to_process = image_path
            else:
                image_to_process = Image.open(image_path)

            # Step 2: Extract with QwenVL async
            print("Extracting with QwenVL...")
            extraction_results = await self.extractor.extract_async(
                [image_to_process],
                include_images=include_images,
                include_usage=include_usage
            )
            
            # Get the first (and only) result
            extraction_result = extraction_results[0]

            if not extraction_result['success']:
                return extraction_result

            # Save QwenVL output if output_dir specified
            if self.output_dir:
                self._save_output(
                    content=extraction_result['markdown'],
                    image_path=image_path,
                    stage="0_qwen_original"
                )

            # Step 3: Convert tables to HTML (if enabled)
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

            return extraction_result

        except Exception as e:
            return {
                'success': False,
                'markdown': '',
                'images': [],
                'elements': [],
                'usage': {},
                'error': str(e)
            }

    def _extract_with_retry(
        self,
        image_to_process,
        include_images: bool = True,
        include_usage: bool = False,
        max_retries: int = 3,
        initial_delay: float = 1.0
    ) -> Dict:
        """
        Extract with QwenVL with retry logic and exponential backoff.
        
        Args:
            image_to_process: Image to process
            include_images: Whether to include images in extraction
            include_usage: Whether to include usage data
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
        
        Returns:
            Extraction result
        """
        delay = initial_delay
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                result = self.extractor.extract(
                    image_to_process,
                    include_images=include_images,
                    include_usage=include_usage
                )
                
                if result['success']:
                    if attempt > 0:
                        print(f"Attempt {attempt + 1}: Extraction succeeded")
                    return result
                else:
                    # If extraction failed but no exception, return the result
                    last_error = result.get('error', 'Unknown error')
                    if attempt < max_retries:
                        print(f"[WARNING] Attempt {attempt + 1}: Extraction failed: {last_error}")
                        print(f"   Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay = min(delay * 2.0, 60.0)
                    else:
                        print(f"[ERROR] Extraction failed after {max_retries + 1} attempts")
                        return result
                        
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    print(f"[WARNING] Attempt {attempt + 1}: Extraction failed: {last_error}")
                    print(f"   Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * 2.0, 60.0)
                else:
                    print(f"[ERROR] Extraction failed after {max_retries + 1} attempts: {last_error}")
                    return {
                        'success': False,
                        'markdown': '',
                        'images': [],
                        'elements': [],
                        'usage': {},
                        'error': last_error
                    }
        
        # Fallback (should not reach here)
        return {
            'success': False,
            'markdown': '',
            'images': [],
            'elements': [],
            'usage': {},
            'error': last_error or 'Unknown error'
        }

    def _save_output(self, content: str, image_path, stage: str):
        """
        Save output to file in output directory.

        Args:
            content: Markdown/HTML content to save
            image_path: Original image path (str) or PIL Image object (used for filename)
            stage: Stage name (e.g., "0_qwen_original", "1_iteration_1", "2_iteration_2")
        """
        if not self.output_dir:
            return

        # Create output directory if it doesn't exist
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename: image_name_stage_timestamp.md
        if isinstance(image_path, str):
            image_name = Path(image_path).stem
        else:
            # PIL Image object - use generic name
            image_name = "image"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_name}_{stage}_{timestamp}.md"

        output_file = output_path / filename

        # Save content
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Saved: {output_file}")
