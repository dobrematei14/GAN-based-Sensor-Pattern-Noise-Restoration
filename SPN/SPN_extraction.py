import os
import rawpy  # For reading .DNG files
import numpy as np
from SPN_extraction_methods import save_spn_as_image, extract_compressed_images_spn, extract_spn_wavelet
import cv2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get paths from environment variables
EXTERNAL_DRIVE = os.getenv('EXTERNAL_DRIVE')
SPN_IMAGES_DIR = os.getenv('SPN_IMAGES_DIR')
COMPRESSED_IMAGES_DIR = os.getenv('COMPRESSED_IMAGES_DIR')
ORIGINAL_IMAGES_DIR = os.getenv('ORIGINAL_IMAGES_DIR')

def process_dng_images_in_folders(root_path=None):
    """
    Process all folders in the root path, extract SPN from all .DNG images in each folder,
    average them to create a single camera_SPN, and save it in the appropriate directory.

    Args:
        root_path (str, optional): Path to the root directory containing camera model folders.
                                  If None, uses environment variable.

    Returns:
        dict: Dictionary containing processing results and statistics
    """
    # Use environment variable if root_path is not provided
    if root_path is None:
        root_path = os.path.join(EXTERNAL_DRIVE, ORIGINAL_IMAGES_DIR)

    results = {
        'processed_cameras': [],
        'errors': [],
        'total_images_processed': 0
    }

    # Iterate over all folders in the root path
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")

            # Find all .DNG files in the folder
            dng_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.dng')]
            if not dng_files:
                print(f"No .DNG files found in folder: {folder_name}")
                continue

            # Create the SPN directory structure
            spn_base_dir = os.path.join(EXTERNAL_DRIVE, SPN_IMAGES_DIR, folder_name)
            original_spn_dir = os.path.join(spn_base_dir, 'original')
            os.makedirs(original_spn_dir, exist_ok=True)

            # List to store all SPNs for averaging
            all_spns = []

            # Process each .DNG file
            for dng_file in dng_files:
                dng_path = os.path.join(folder_path, dng_file)

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

                        # Extract SPN using wavelet decomposition
                        spn = extract_spn_wavelet(raw_image_normalized, wavelet='db1', level=1)
                        all_spns.append(spn)
                        results['total_images_processed'] += 1

                        print(f"Extracted SPN from {dng_file}")

                except Exception as e:
                    error_msg = f"Error processing {dng_file}: {e}"
                    print(error_msg)
                    results['errors'].append(error_msg)

            if all_spns:
                # Average all SPNs
                average_spn = np.mean(all_spns, axis=0)
                
                # Normalize the average SPN
                average_spn_normalized = cv2.normalize(average_spn, None, 0, 255, cv2.NORM_MINMAX)
                average_spn_normalized = np.uint8(average_spn_normalized)

                # Save the average SPN as camera_SPN
                camera_spn_path = os.path.join(original_spn_dir, 'camera_SPN.png')
                cv2.imwrite(camera_spn_path, average_spn_normalized)
                print(f"Saved average camera SPN for {folder_name} in {original_spn_dir}")
                results['processed_cameras'].append(folder_name)

    return results

def process_compressed_images():
    """
    Process compressed images and extract individual SPNs for each image.

    Returns:
        dict: Dictionary containing processing results and statistics
    """
    compressed_base_path = os.path.join(EXTERNAL_DRIVE, COMPRESSED_IMAGES_DIR)
    spn_base_path = os.path.join(EXTERNAL_DRIVE, SPN_IMAGES_DIR)

    results = {
        'processed_images': [],
        'skipped_images': [],
        'errors': [],
        'total_images_processed': 0
    }

    # Get list of camera models
    camera_models = [d for d in os.listdir(compressed_base_path)
                    if os.path.isdir(os.path.join(compressed_base_path, d))]

    for camera_model in camera_models:
        camera_dir = os.path.join(compressed_base_path, camera_model)
        
        # Get list of quality levels
        quality_levels = [q for q in os.listdir(camera_dir)
                         if os.path.isdir(os.path.join(camera_dir, q))]

        for quality_level in quality_levels:
            quality_dir = os.path.join(camera_dir, quality_level)
            
            # Create output directory for this quality level
            output_dir = os.path.join(spn_base_path, camera_model, 'compressed', quality_level)
            os.makedirs(output_dir, exist_ok=True)

            # Get all compressed images in this quality level
            compressed_images = [f for f in os.listdir(quality_dir)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for image_file in compressed_images:
                image_path = os.path.join(quality_dir, image_file)
                output_filename = f"{os.path.splitext(image_file)[0]}_SPN.png"
                output_path = os.path.join(output_dir, output_filename)

                # Skip if the SPN already exists
                if os.path.exists(output_path):
                    print(f"SPN already exists for {image_file}, skipping")
                    results['skipped_images'].append(image_path)
                    continue

                try:
                    # Extract and save the SPN
                    save_spn_as_image(
                        image_path=image_path,
                        output_path=output_dir,
                        wavelet='db1',
                        level=1,
                        camera_model=os.path.splitext(image_file)[0],
                        format='png'
                    )
                    print(f"Extracted and saved SPN for {image_file}")
                    results['processed_images'].append(image_path)
                    results['total_images_processed'] += 1

                except Exception as e:
                    error_msg = f"Error processing {image_path}: {e}"
                    print(error_msg)
                    results['errors'].append(error_msg)

    return results

def extract_all_spns():
    """
    Extract SPNs from both original and compressed images.
    This is the main function to call for the complete SPN extraction pipeline.

    Returns:
        dict: Dictionary containing combined results from both processing steps
    """
    print("Starting SPN extraction pipeline...")
    
    # Process original images and create average camera SPNs
    original_results = process_dng_images_in_folders()
    print("\nOriginal images processing completed.")
    
    # Process compressed images and create individual SPNs
    compressed_results = process_compressed_images()
    print("\nCompressed images processing completed.")
    
    # Combine results
    combined_results = {
        'original_processing': original_results,
        'compressed_processing': compressed_results,
        'total_images_processed': original_results['total_images_processed'] + compressed_results['total_images_processed']
    }
    
    print(f"\nSPN extraction completed. Total images processed: {combined_results['total_images_processed']}")
    return combined_results