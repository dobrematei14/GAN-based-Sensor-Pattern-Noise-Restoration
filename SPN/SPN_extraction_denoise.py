import os
import rawpy
import numpy as np
import cv2
from dotenv import load_dotenv
from tqdm import tqdm
import random
from .SPN_extraction_methods import extract_spn_denoise_wavelet

# Load environment variables
load_dotenv()

# Get paths from environment variables
EXTERNAL_DRIVE = os.getenv('EXTERNAL_DRIVE')
SPN_IMAGES_DIR = os.getenv('SPN_IMAGES_DIR')
COMPRESSED_IMAGES_DIR = os.getenv('COMPRESSED_IMAGES_DIR')
ORIGINAL_IMAGES_DIR = os.getenv('ORIGINAL_IMAGES_DIR')

def save_spn_denoise_as_image(image_path, output_path, wavelet='db8', camera_model=None, format='png'):
    """
    Extract SPN using denoise wavelet method and save it as an image file.
    
    Args:
        image_path (str): Path to the input image
        output_path (str): Directory to save the SPN image
        wavelet (str, optional): Wavelet type to use. Defaults to 'db8'.
        camera_model (str, optional): Camera model name. Defaults to None.
        format (str, optional): Output image format. Defaults to 'png'.
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract SPN
    spn = extract_spn_denoise_wavelet(img, wavelet=wavelet, is_color=False)
    
    # Normalize to 0-255 range
    spn_normalized = cv2.normalize(spn, None, 0, 255, cv2.NORM_MINMAX)
    spn_normalized = np.uint8(spn_normalized)
    
    # Generate output filename
    if camera_model:
        output_filename = f"{camera_model}_SPN.{format}"
    else:
        output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_SPN.{format}"
    
    # Save the SPN
    output_filepath = os.path.join(output_path, output_filename)
    cv2.imwrite(output_filepath, spn_normalized)

def process_raw_images_in_folders(root_path=None):
    """
    Process all folders in the root path, extract SPN from one RAW image (DNG or RW2) in each folder,
    using the denoise wavelet method. Shows progress bars for camera models.

    Args:
        root_path (str, optional): Path to the root directory containing camera model folders.
                                  If None, uses environment variable.

    Returns:
        dict: Dictionary containing processing results and statistics
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
        'selected_images': {}
    }

    camera_folders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    with tqdm(total=len(camera_folders), desc="Processing RAW images", unit="camera", position=0, leave=True) as pbar:
        for folder_name in camera_folders:
            folder_path = os.path.join(root_path, folder_name)
            spn_base_dir = os.path.join(EXTERNAL_DRIVE, SPN_IMAGES_DIR, folder_name)
            original_spn_dir = os.path.join(spn_base_dir, 'original')
            camera_spn_path = os.path.join(original_spn_dir, 'camera_SPN.png')
            
            if os.path.exists(camera_spn_path):
                results['skipped_cameras'].append(folder_name)
                raw_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.dng', '.rw2'))]
                results['total_images_skipped'] += len(raw_files)
                pbar.update(1)
                continue

            raw_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.dng', '.rw2'))]
            
            if not raw_files:
                warning = f"Warning: No RAW images found for {folder_name}. Skipping this camera."
                results['warnings'].append(warning)
                pbar.update(1)
                continue

            selected_file = random.choice(raw_files)
            os.makedirs(original_spn_dir, exist_ok=True)

            raw_path = os.path.join(folder_path, selected_file)
            try:
                with rawpy.imread(raw_path) as raw:
                    raw_image = raw.raw_image
                    if len(raw_image.shape) == 3:
                        raw_image = np.mean(raw_image, axis=2)

                    raw_image_normalized = cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX)
                    raw_image_normalized = np.uint8(raw_image_normalized)

                    # Extract SPN using denoise wavelet method
                    spn = extract_spn_denoise_wavelet(raw_image_normalized, wavelet='db8', is_color=False)
                    
                    spn_normalized = cv2.normalize(spn, None, 0, 255, cv2.NORM_MINMAX)
                    spn_normalized = np.uint8(spn_normalized)

                    cv2.imwrite(camera_spn_path, spn_normalized)
                    
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
    Process compressed images and extract individual SPNs using denoise wavelet method.
    Shows progress bars for camera models and individual image processing.

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

    camera_models = [d for d in os.listdir(compressed_base_path)
                    if os.path.isdir(os.path.join(compressed_base_path, d))]

    with tqdm(total=len(camera_models), desc="Processing compressed images", unit="camera", position=0, leave=True) as camera_pbar:
        for camera_model in camera_models:
            camera_dir = os.path.join(compressed_base_path, camera_model)
            quality_levels = [q for q in os.listdir(camera_dir)
                            if os.path.isdir(os.path.join(camera_dir, q))]

            for quality_level in quality_levels:
                quality_dir = os.path.join(camera_dir, quality_level)
                output_dir = os.path.join(spn_base_path, camera_model, 'compressed', quality_level)
                os.makedirs(output_dir, exist_ok=True)

                compressed_images = [f for f in os.listdir(quality_dir)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                if not compressed_images:
                    continue

                with tqdm(total=len(compressed_images), 
                         desc=f"Processing {camera_model} {quality_level}%", 
                         unit="image",
                         position=1, 
                         leave=False) as image_pbar:
                    for image_file in compressed_images:
                        image_path = os.path.join(quality_dir, image_file)
                        output_filename = f"{os.path.splitext(image_file)[0]}_SPN.png"
                        output_path = os.path.join(output_dir, output_filename)

                        if os.path.exists(output_path):
                            results['skipped_images'].append(image_path)
                            image_pbar.update(1)
                            continue

                        try:
                            save_spn_denoise_as_image(
                                image_path=image_path,
                                output_path=output_dir,
                                wavelet='db8',
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

def extract_all_spns_denoise():
    """
    Extract SPNs from both original and compressed images using denoise wavelet method.
    Shows progress bars and detailed processing statistics.

    Returns:
        dict: Dictionary containing combined results from both processing steps
    """
    print("Starting SPN extraction pipeline (Denoise Wavelet Method)...")
    
    original_results = process_raw_images_in_folders()
    print("\nOriginal images processing completed.")
    print(f"- Cameras processed: {len(original_results['processed_cameras'])}")
    print(f"- Cameras skipped: {len(original_results['skipped_cameras'])}")
    
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
    
    compressed_results = process_compressed_images()
    print("\nCompressed images processing completed.")
    print(f"- Images processed: {compressed_results['total_images_processed']}")
    print(f"- Images skipped: {len(compressed_results['skipped_images'])}")
    if compressed_results['errors']:
        print("\nErrors during compressed processing:")
        for error in compressed_results['errors']:
            print(f"- {error}")
    
    combined_results = {
        'original_processing': original_results,
        'compressed_processing': compressed_results,
        'total_images_processed': original_results['total_images_processed'] + compressed_results['total_images_processed']
    }
    
    print(f"\nSPN extraction completed. Total images processed: {combined_results['total_images_processed']}")
    return combined_results 