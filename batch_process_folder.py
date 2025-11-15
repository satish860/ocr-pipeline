"""
Batch process all images in a folder using QwenVL markdown extraction.

This script processes all images in a specified folder and saves the results
in organized output directories.
"""

import os
import sys
from pathlib import Path
from test_qwen_html_parser import main as process_image

def batch_process_folder(input_folder: str):
    """
    Process all images in a folder.

    Args:
        input_folder: Path to folder containing images
    """
    input_path = Path(input_folder)

    if not input_path.exists():
        print(f"Error: Folder '{input_folder}' does not exist")
        return

    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))

    if not image_files:
        print(f"No image files found in '{input_folder}'")
        return

    # Remove duplicates and sort files for consistent processing order
    image_files = sorted(set(image_files))

    print("=" * 80)
    print(f"Batch Processing: {len(image_files)} images")
    print("=" * 80)
    print(f"Input folder: {input_folder}")
    print(f"Files to process:")
    for i, img in enumerate(image_files, 1):
        print(f"  {i:2d}. {img.name}")
    print()

    # Clear output folder once at the start
    output_dir = Path("output")
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        print("Cleared output folder\n")
    output_dir.mkdir(exist_ok=True)

    # Process each image
    successful = 0
    failed = 0

    for i, image_file in enumerate(image_files, 1):
        print("\n" + "=" * 80)
        print(f"Processing {i}/{len(image_files)}: {image_file.name}")
        print("=" * 80)

        try:
            # Process the image (don't clear output folder for each image)
            process_image(str(image_file), clear_output=False)
            successful += 1
            print(f"\n[OK] Successfully processed: {image_file.name}")
        except Exception as e:
            failed += 1
            print(f"\n[ERROR] Processing {image_file.name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total files: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nAll results saved to: output/")


if __name__ == "__main__":
    # Default folder or from command line
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "input/2e1b63c5-761d-48b9-b3b5-f263c3db4e30_images"

    batch_process_folder(folder)
