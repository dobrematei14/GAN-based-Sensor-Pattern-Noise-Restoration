import os
import rawpy  # For reading RAW files (DNG and RW2)
import numpy as np
from .SPN_extraction_methods import save_spn, extract_spn
import cv2
from dotenv import load_dotenv
from tqdm import tqdm
import random
from .SPN_extraction_original import process_raw

# Load environment variables
load_dotenv()

# Get paths from environment variables
DRIVE = os.getenv('EXTERNAL_DRIVE')
SPN_DIR = os.getenv('SPN_IMAGES_DIR')
COMP_DIR = os.getenv('COMPRESSED_IMAGES_DIR')
ORIG_DIR = os.getenv('ORIGINAL_IMAGES_DIR')

def process_compressed():
    """Process compressed images and extract SPNs."""
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
                        save_spn(
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
    """Extract SPNs from both original and compressed images."""
    print("Starting SPN extraction pipeline...")
    
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