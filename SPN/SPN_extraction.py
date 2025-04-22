import os
import rawpy  # For reading RAW files (DNG and RW2)
import numpy as np
from .SPN_extraction_methods import save_spn_as_image, extract_spn_wavelet
import cv2
from dotenv import load_dotenv
from tqdm import tqdm
import random

# Load environment variables
load_dotenv()

# Get paths from environment variables
EXTERNAL_DRIVE = os.getenv('EXTERNAL_DRIVE')
SPN_IMAGES_DIR = os.getenv('SPN_IMAGES_DIR')
COMPRESSED_IMAGES_DIR = os.getenv('COMPRESSED_IMAGES_DIR')
ORIGINAL_IMAGES_DIR = os.getenv('ORIGINAL_IMAGES_DIR')

def process_raw_images_in_folders(root_path=None):
    """
    Process all folders in the root path, extract SPN from one RAW image (DNG or RW2) in each folder,
    and save it as the camera_SPN. The image is randomly selected from all available RAW images.
    Shows progress bars for camera models.

    Args:
        root_path (str, optional): Path to the root directory containing camera model folders.
                                  If None, uses environment variable.

    Returns:
        dict: Dictionary containing processing results and statistics:
            - processed_cameras: List of successfully processed camera directories
            - skipped_cameras: List of skipped camera directories
            - errors: List of any errors encountered
            - warnings: List of any warnings encountered
            - total_images_processed: Total count of processed images
            - total_images_skipped: Total count of skipped images
            - selected_images: Dictionary mapping camera models to their selected RAW image
    """
    if root_path is None:
        root_path = os.path.join(EXTERNAL_DRIVE, ORIGINAL_IMAGES_DIR)

    results = {
        'processed_cameras': [],
        'skipped_cameras': [],
        'errors': [],
        'warnings': [],
        'total_images_processed': 0,
        'total_images_skipped': 0,
        'selected_images': {}  # Track which image was used for each camera
    }

    # Get list of all folders first
    camera_folders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    # Process each camera model
    with tqdm(total=len(camera_folders), desc="Processing RAW images", unit="camera", position=0, leave=True) as pbar:
        for folder_name in camera_folders:
            folder_path = os.path.join(root_path, folder_name)

            # Create the SPN directory structure and check if already processed
            spn_base_dir = os.path.join(EXTERNAL_DRIVE, SPN_IMAGES_DIR, folder_name)
            original_spn_dir = os.path.join(spn_base_dir, 'original')
            camera_spn_path = os.path.join(original_spn_dir, 'camera_SPN.png')
            
            # Skip if camera_SPN already exists
            if os.path.exists(camera_spn_path):
                results['skipped_cameras'].append(folder_name)
                raw_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.dng', '.rw2'))]
                results['total_images_skipped'] += len(raw_files)
                pbar.update(1)
                continue

            # Find all RAW files in the folder
            raw_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.dng', '.rw2'))]
            
            # Check if we have any images
            if not raw_files:
                warning = f"Warning: No RAW images found for {folder_name}. Skipping this camera."
                results['warnings'].append(warning)
                pbar.update(1)
                continue

            # Randomly select one image
            selected_file = random.choice(raw_files)

            # Create directory if needed
            os.makedirs(original_spn_dir, exist_ok=True)

            # Process the selected RAW file
            raw_path = os.path.join(folder_path, selected_file)
            try:
                with rawpy.imread(raw_path) as raw:
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
                    
                    # Normalize the SPN
                    spn_normalized = cv2.normalize(spn, None, 0, 255, cv2.NORM_MINMAX)
                    spn_normalized = np.uint8(spn_normalized)

                    # Save the SPN
                    cv2.imwrite(camera_spn_path, spn_normalized)
                    
                    # Update results
                    results['processed_cameras'].append(folder_name)
                    results['total_images_processed'] += 1
                    results['selected_images'][folder_name] = selected_file

            except Exception as e:
                error_msg = f"Error processing {selected_file}: {e}"
                results['errors'].append(error_msg)
            
            pbar.update(1)

    return results

def process_compressed_images():
    """
    Process compressed images and extract individual SPNs for each image.
    Shows progress bars for camera models and individual image processing.

    Returns:
        dict: Dictionary containing processing results and statistics:
            - processed_images: List of successfully processed images
            - skipped_images: List of skipped images
            - errors: List of any errors encountered
            - total_images_processed: Total count of processed images
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

    # Process each camera model
    with tqdm(total=len(camera_models), desc="Processing compressed images", unit="camera", position=0, leave=True) as camera_pbar:
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

                if not compressed_images:
                    continue

                # Process images sequentially
                with tqdm(total=len(compressed_images), 
                         desc=f"Processing {camera_model} {quality_level}%", 
                         unit="image",
                         position=1, 
                         leave=False) as image_pbar:
                    for image_file in compressed_images:
                        image_path = os.path.join(quality_dir, image_file)
                        output_filename = f"{os.path.splitext(image_file)[0]}_SPN.png"
                        output_path = os.path.join(output_dir, output_filename)

                        # Skip if the SPN already exists
                        if os.path.exists(output_path):
                            results['skipped_images'].append(image_path)
                            image_pbar.update(1)
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
                            results['processed_images'].append(image_path)
                            results['total_images_processed'] += 1

                        except Exception as e:
                            error_msg = f"Error processing {image_path}: {e}"
                            results['errors'].append(error_msg)
                        
                        image_pbar.update(1)
            
            camera_pbar.update(1)

    return results

def extract_all_spns():
    """
    Extract SPNs from both original and compressed images.
    This is the main function to call for the complete SPN extraction pipeline.
    Shows progress bars and detailed processing statistics.

    Returns:
        dict: Dictionary containing combined results from both processing steps:
            - original_processing: Results from RAW image processing
            - compressed_processing: Results from compressed image processing
            - total_images_processed: Total count of processed images
    """
    print("Starting SPN extraction pipeline...")
    
    # Process original images and create camera SPNs
    original_results = process_raw_images_in_folders()
    print("\nOriginal images processing completed.")
    print(f"- Cameras processed: {len(original_results['processed_cameras'])}")
    print(f"- Cameras skipped: {len(original_results['skipped_cameras'])}")
    
    # Print which image was used for each camera
    if original_results['selected_images']:
        print("\nSelected images for camera SPNs:")
        for camera, image in original_results['selected_images'].items():
            print(f"- {camera}: {image}")
    
    if original_results['warnings']:
        print("\nWarnings during processing:")
        for warning in original_results['warnings']:
            print(f"- {warning}")
            
    if original_results['errors']:
        print("\nErrors during RAW processing:")
        for error in original_results['errors']:
            print(f"- {error}")
    
    # Process compressed images and create individual SPNs
    compressed_results = process_compressed_images()
    print("\nCompressed images processing completed.")
    print(f"- Images processed: {compressed_results['total_images_processed']}")
    print(f"- Images skipped: {len(compressed_results['skipped_images'])}")
    if compressed_results['errors']:
        print("\nErrors during compressed processing:")
        for error in compressed_results['errors']:
            print(f"- {error}")
    
    # Combine results
    combined_results = {
        'original_processing': original_results,
        'compressed_processing': compressed_results,
        'total_images_processed': original_results['total_images_processed'] + compressed_results['total_images_processed']
    }
    
    print(f"\nSPN extraction completed. Total images processed: {combined_results['total_images_processed']}")
    return combined_results