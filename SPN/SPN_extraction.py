import os
import rawpy  # For reading .DNG files
import numpy as np
from SPN_extraction_methods import save_spn_as_image, extract_compressed_images_spn
import cv2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get paths from environment variables
EXTERNAL_DRIVE = os.getenv('EXTERNAL_DRIVE')
SPN_IMAGES_DIR = os.getenv('SPN_IMAGES_DIR')
COMPRESSED_IMAGES_DIR = os.getenv('COMPRESSED_IMAGES_DIR')

def process_dng_images_in_folders(root_path=None):
    """
    Process all folders in the root path, extract SPN from the .DNG image in each folder,
    and save the SPN as a PNG image in a subfolder named 'SPN'.

    Args:
        root_path (str, optional): Path to the root directory containing camera model folders.
                                  If None, uses environment variable.
    """
    # Use environment variable if root_path is not provided
    if root_path is None:
        root_path = os.path.join(EXTERNAL_DRIVE, SPN_IMAGES_DIR)

    # Iterate over all folders in the root path
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")

            # Find the .DNG file in the folder (assuming there's only one image)
            dng_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.dng')]
            if not dng_files:
                print(f"No .DNG file found in folder: {folder_name}")
                continue

            # Use the first .DNG file found (assuming only one image per folder)
            dng_file = dng_files[0]
            dng_path = os.path.join(folder_path, dng_file)

            # Create the SPN subfolder if it doesn't exist
            spn_folder = os.path.join(folder_path, 'SPN')
            os.makedirs(spn_folder, exist_ok=True)

            # Read the .DNG file and extract the raw image data
            try:
                with rawpy.imread(dng_path) as raw:
                    # Extract the raw image data (Bayer pattern)
                    raw_image = raw.raw_image

                    # Convert to grayscale by averaging the color channels
                    if len(raw_image.shape) == 3:  # If the image has multiple channels
                        raw_image = np.mean(raw_image, axis=2)

                    # Normalize the raw image to [0, 255] for SPN extraction
                    raw_image_normalized = cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX)
                    raw_image_normalized = np.uint8(raw_image_normalized)

                    # Save the normalized raw image temporarily (optional, for debugging)
                    temp_image_path = os.path.join(spn_folder, 'temp_raw_image.png')
                    cv2.imwrite(temp_image_path, raw_image_normalized)

                    # Extract and save the SPN
                    save_spn_as_image(
                        image_path=temp_image_path,  # Use the normalized raw image
                        output_path=spn_folder,
                        wavelet='db1',
                        level=1,
                        camera_model=folder_name,  # Use folder name as camera model
                        format='png'
                    )
                    print(f"SPN saved for {folder_name} in {spn_folder}")

                    # Clean up: Remove the temporary raw image
                    os.remove(temp_image_path)
            except Exception as e:
                print(f"Error processing {folder_name}: {e}")


if __name__ == "__main__":
    # Use environment variables for paths
    root_path = os.path.join(EXTERNAL_DRIVE, SPN_IMAGES_DIR)
    process_dng_images_in_folders(root_path)

    compressed_base_path = os.path.join(EXTERNAL_DRIVE, COMPRESSED_IMAGES_DIR)
    output_base_path = os.path.join(EXTERNAL_DRIVE, SPN_IMAGES_DIR)

    extract_compressed_images_spn(compressed_base_path, output_base_path)