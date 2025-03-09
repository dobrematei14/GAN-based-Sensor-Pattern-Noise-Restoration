import os
import rawpy
import imageio
from PIL import Image

# Path to the directory containing the images
path = "Images/Original"
path_compressed = "Images/Compressed"
quality_levels = [90, 60, 30]  # Different JPEG quality levels

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
                image.save(output_path, "JPEG", quality=quality)