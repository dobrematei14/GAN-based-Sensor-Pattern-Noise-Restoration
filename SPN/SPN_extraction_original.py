import os
import rawpy
import numpy as np
import cv2
from dotenv import load_dotenv
from tqdm import tqdm
import random
from .SPN_extraction_methods import extract_spn

# Load environment variables
load_dotenv()

# Get paths from environment variables
DRIVE = os.getenv('EXTERNAL_DRIVE')
SPN_DIR = os.getenv('SPN_IMAGES_DIR')
ORIG_DIR = os.getenv('ORIGINAL_IMAGES_DIR')

# Number of images to process per camera
IMAGES_PER_CAMERA = 200

def process_raw(root_path=None):
    """Process RAW images and extract SPNs for each camera.
    
    For each camera, this function:
    1. Selects up to IMAGES_PER_CAMERA (200) random RAW images
    2. Extracts the SPN from each selected image
    3. Averages all extracted SPNs to create a robust camera fingerprint
    4. Saves the averaged SPN as the camera's fingerprint
    
    Args:
        root_path (str, optional): Path to the directory containing camera folders.
            If None, uses the path from environment variables.
    
    Returns:
        dict: Processing results containing:
            - processed: List of successfully processed cameras
            - skipped: List of cameras skipped (already processed)
            - errors: List of error messages
            - warnings: List of warning messages
            - total_processed: Total number of images processed
            - total_skipped: Total number of images skipped
            - selected: Dictionary mapping cameras to their selected image files
    """
    if root_path is None:
        root_path = os.path.join(DRIVE, ORIG_DIR)

    results = {
        'processed': [],
        'skipped': [],
        'errors': [],
        'warnings': [],
        'total_processed': 0,
        'total_skipped': 0,
        'selected': {}
    }

    cams = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    with tqdm(total=len(cams), desc="Processing RAW images", unit="camera") as pbar:
        for cam in cams:
            cam_path = os.path.join(root_path, cam)
            spn_dir = os.path.join(DRIVE, SPN_DIR, cam)
            orig_dir = os.path.join(spn_dir, 'original')
            spn_path = os.path.join(orig_dir, 'camera_SPN.png')
            
            if os.path.exists(spn_path):
                results['skipped'].append(cam)
                raw_files = [f for f in os.listdir(cam_path) if f.lower().endswith(('.dng', '.rw2'))]
                results['total_skipped'] += len(raw_files)
                pbar.update(1)
                continue

            raw_files = [f for f in os.listdir(cam_path) if f.lower().endswith(('.dng', '.rw2'))]
            
            if not raw_files:
                results['warnings'].append(f"No RAW images found for {cam}")
                pbar.update(1)
                continue

            # Select up to IMAGES_PER_CAMERA images
            selected_files = random.sample(raw_files, min(IMAGES_PER_CAMERA, len(raw_files)))
            os.makedirs(orig_dir, exist_ok=True)

            try:
                first_shape = None
                used_files = []
                running_sum = None
                count = 0
                
                for selected in selected_files:
                    print(f"Processing {cam}: {selected}")
                    with rawpy.imread(os.path.join(cam_path, selected)) as raw:
                        img = raw.raw_image
                        if len(img.shape) == 3:
                            img = np.mean(img, axis=2)

                        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                        img_norm = np.uint8(img_norm)

                        spn = extract_spn(img_norm)
                        
                        # Store the shape of the first SPN
                        if first_shape is None:
                            first_shape = spn.shape
                            running_sum = spn.astype(np.float64)
                            count = 1
                            used_files.append(selected)
                        # Only use SPNs that match the first one's shape
                        elif spn.shape == first_shape:
                            running_sum += spn.astype(np.float64)
                            count += 1
                            used_files.append(selected)
                        else:
                            print(f"Skipping {selected} - different size: {spn.shape} vs {first_shape}")

                if count == 0:
                    results['errors'].append(f"No valid SPNs found for {cam}")
                    continue

                # Calculate final average
                avg_spn = running_sum / count
                spn_norm = cv2.normalize(avg_spn, None, 0, 255, cv2.NORM_MINMAX)
                spn_norm = np.uint8(spn_norm)

                cv2.imwrite(spn_path, spn_norm)
                
                results['processed'].append(cam)
                results['total_processed'] += len(used_files)
                results['selected'][cam] = used_files

            except Exception as e:
                results['errors'].append(f"Error processing camera {cam}: {e}")
            
            pbar.update(1)

    return results 