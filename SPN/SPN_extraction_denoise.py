import os
import rawpy
import numpy as np
import cv2
from dotenv import load_dotenv
from tqdm import tqdm
import random
from .SPN_extraction_methods import extract_spn_denoise

# Load environment variables
load_dotenv()

# Get paths from environment variables
DRIVE = os.getenv('EXTERNAL_DRIVE')
SPN_DIR = os.getenv('SPN_IMAGES_DIR')
COMP_DIR = os.getenv('COMPRESSED_IMAGES_DIR')
ORIG_DIR = os.getenv('ORIGINAL_IMAGES_DIR')

def save_spn_denoise(img_path, out_path, wavelet='db8', cam=None, fmt='png'):
    """
    Extract SPN using denoise wavelet method and save it as an image file.
    
    Args:
        img_path (str): Path to the input image
        out_path (str): Directory to save the SPN image
        wavelet (str, optional): Wavelet type to use. Defaults to 'db8'.
        cam (str, optional): Camera model name. Defaults to None.
        fmt (str, optional): Output image format. Defaults to 'png'.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    spn = extract_spn_denoise(img, wavelet=wavelet)
    spn_norm = cv2.normalize(spn, None, 0, 255, cv2.NORM_MINMAX)
    spn_norm = np.uint8(spn_norm)
    
    out_name = f"{cam}_SPN.{fmt}" if cam else f"{os.path.splitext(os.path.basename(img_path))[0]}_SPN.{fmt}"
    cv2.imwrite(os.path.join(out_path, out_name), spn_norm)

def process_raw(root_path=None):
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

            selected = random.choice(raw_files)
            os.makedirs(orig_dir, exist_ok=True)

            try:
                with rawpy.imread(os.path.join(cam_path, selected)) as raw:
                    img = raw.raw_image
                    if len(img.shape) == 3:
                        img = np.mean(img, axis=2)

                    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                    img_norm = np.uint8(img_norm)

                    spn = extract_spn_denoise(img_norm)
                    spn_norm = cv2.normalize(spn, None, 0, 255, cv2.NORM_MINMAX)
                    spn_norm = np.uint8(spn_norm)

                    cv2.imwrite(spn_path, spn_norm)
                    
                    results['processed'].append(cam)
                    results['total_processed'] += 1
                    results['selected'][cam] = selected

            except Exception as e:
                results['errors'].append(f"Error processing {selected}: {e}")
            
            pbar.update(1)

    return results

def process_compressed():
    """
    Process compressed images and extract individual SPNs using denoise wavelet method.
    Shows progress bars for camera models and individual image processing.

    Returns:
        dict: Dictionary containing processing results and statistics
    """
    comp_path = os.path.join(DRIVE, COMP_DIR)
    spn_path = os.path.join(DRIVE, SPN_DIR)

    results = {
        'processed': [],
        'skipped': [],
        'errors': [],
        'total_processed': 0
    }

    cams = [d for d in os.listdir(comp_path) if os.path.isdir(os.path.join(comp_path, d))]

    with tqdm(total=len(cams), desc="Processing compressed images", unit="camera") as pbar:
        for cam in cams:
            cam_dir = os.path.join(comp_path, cam)
            qualities = [q for q in os.listdir(cam_dir) if os.path.isdir(os.path.join(cam_dir, q))]

            for q in qualities:
                q_dir = os.path.join(cam_dir, q)
                out_dir = os.path.join(spn_path, cam, 'compressed', q)
                os.makedirs(out_dir, exist_ok=True)

                imgs = [f for f in os.listdir(q_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                if not imgs:
                    continue

                for img in imgs:
                    img_path = os.path.join(q_dir, img)
                    out_name = f"{os.path.splitext(img)[0]}_SPN.png"
                    out_path = os.path.join(out_dir, out_name)

                    if os.path.exists(out_path):
                        results['skipped'].append(img_path)
                        continue

                    try:
                        save_spn_denoise(
                            img_path=img_path,
                            out_path=out_dir,
                            cam=os.path.splitext(img)[0]
                        )
                        results['processed'].append(img_path)
                        results['total_processed'] += 1

                    except Exception as e:
                        results['errors'].append(f"Error processing {img_path}: {e}")
            
            pbar.update(1)

    return results

def extract_all():
    """
    Extract SPNs from both original and compressed images using denoise wavelet method.
    Shows progress bars and detailed processing statistics.

    Returns:
        dict: Dictionary containing combined results from both processing steps
    """
    print("Starting SPN extraction pipeline (Denoise Wavelet Method)...")
    
    orig_results = process_raw()
    print("\nOriginal images processing completed.")
    print(f"- Cameras processed: {len(orig_results['processed'])}")
    print(f"- Cameras skipped: {len(orig_results['skipped'])}")
    
    if orig_results['selected']:
        print("\nSelected images for camera SPNs:")
        for cam, img in orig_results['selected'].items():
            print(f"- {cam}: {img}")
    
    if orig_results['warnings']:
        print("\nWarnings during processing:")
        for warning in orig_results['warnings']:
            print(f"- {warning}")
            
    if orig_results['errors']:
        print("\nErrors during RAW processing:")
        for error in orig_results['errors']:
            print(f"- {error}")
    
    comp_results = process_compressed()
    print("\nCompressed images processing completed.")
    print(f"- Images processed: {comp_results['total_processed']}")
    print(f"- Images skipped: {len(comp_results['skipped'])}")
    
    if comp_results['errors']:
        print("\nErrors during compressed processing:")
        for error in comp_results['errors']:
            print(f"- {error}")
    
    return {
        'original': orig_results,
        'compressed': comp_results,
        'total_processed': orig_results['total_processed'] + comp_results['total_processed']
    } 