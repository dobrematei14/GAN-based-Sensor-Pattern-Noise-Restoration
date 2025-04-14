import os
import rawpy
import imageio
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get paths from environment variables
external_drive = os.getenv('EXTERNAL_DRIVE')
original_dir = os.getenv('ORIGINAL_IMAGES_DIR')
compressed_dir = os.getenv('COMPRESSED_IMAGES_DIR')

# Construct full paths
path = os.path.join(external_drive, original_dir)
path_compressed = os.path.join(external_drive, compressed_dir)

# Parse quality levels from environment variable
quality_levels = [int(q) for q in os.getenv('QUALITY_LEVELS').split(',')]

# Check if the external drive is connected
if not os.path.exists(external_drive):
    print(f"Error: External drive {external_drive} not found. Please connect the drive and try again.")
    exit(1)

for directory in os.listdir(path):
    # Skip if not a directory
    dir_path = os.path.join(path, directory)
    if not os.path.isdir(dir_path):
        continue

    print(f"Processing directory: {directory}")

    # Process each DNG file in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith(".DNG"):
            input_path = os.path.join(dir_path, filename)

            # Check if this image has already been processed for all quality levels
            already_processed = True
            for quality in quality_levels:
                quality_dir = os.path.join(path_compressed, directory, str(quality))
                output_path = os.path.join(quality_dir, filename[:-4] + ".jpg")
                if not os.path.exists(output_path):
                    already_processed = False
                    break

            # Skip if the image has already been processed for all quality levels
            if already_processed:
                print(f"Skipping already processed image: {filename}")
                continue

            # Convert DNG to RGB using rawpy
            with rawpy.imread(input_path) as raw:
                rgb = raw.postprocess()

            # Create PIL Image for compression
            image = Image.fromarray(rgb)

            # Compress at different quality levels
            for quality in quality_levels:
                # Create quality-specific directory
                quality_dir = os.path.join(path_compressed, directory, str(quality))
                os.makedirs(quality_dir, exist_ok=True)

                # Save compressed image
                output_path = os.path.join(quality_dir, filename[:-4] + ".jpg")

                # Check if this specific quality version already exists
                if not os.path.exists(output_path):
                    image.save(output_path, "JPEG", quality=quality)
                    print(f"Saved {output_path}")
                else:
                    print(f"Skipping existing file: {output_path}")
