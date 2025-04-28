import os
import rawpy
import imageio
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Get paths from environment variables
DRIVE = os.getenv('EXTERNAL_DRIVE')
ORIG_DIR = os.getenv('ORIGINAL_IMAGES_DIR')
COMP_DIR = os.getenv('COMPRESSED_IMAGES_DIR')

# Parse quality levels from environment variable
QUALITIES = [int(q) for q in os.getenv('QUALITY_LEVELS').split(',')]

def check_drive():
    """Checks if the external drive is connected and accessible."""
    if not os.path.exists(DRIVE):
        return False
    return True

def process_dng(img_path, out_dir, qualities):
    """Processes a DNG image and creates compressed JPEG versions at different quality levels."""
    results = {
        'processed': [],
        'skipped': [],
        'errors': []
    }
    
    filename = os.path.basename(img_path)
    directory = os.path.basename(os.path.dirname(img_path))
    
    out_path = os.path.join(out_dir, directory)
    os.makedirs(out_path, exist_ok=True)
    
    try:
        # Check if the highest quality version exists
        highest_q = max(qualities)
        highest_q_dir = os.path.join(out_path, str(highest_q))
        highest_q_file = os.path.join(highest_q_dir, f"{os.path.splitext(filename)[0]}.jpg")
        
        if os.path.exists(highest_q_file):
            results['skipped'].extend(qualities)
            return results
        
        with rawpy.imread(img_path) as raw:
            rgb = raw.postprocess()
            img = Image.fromarray(rgb)
            
            for q in qualities:
                q_dir = os.path.join(out_path, str(q))
                os.makedirs(q_dir, exist_ok=True)
                
                out_file = os.path.join(q_dir, f"{os.path.splitext(filename)[0]}.jpg")
                img.save(out_file, 'JPEG', quality=q)
                results['processed'].append(q)
                
    except Exception as e:
        results['errors'].append(f"Error processing {img_path}: {str(e)}")
    
    return results

def process_rw2(img_path, out_dir, qualities):
    """Processes a RW2 image and creates compressed JPEG versions at different quality levels."""
    results = {
        'processed': [],
        'skipped': [],
        'errors': []
    }
    
    filename = os.path.basename(img_path)
    directory = os.path.basename(os.path.dirname(img_path))
    
    try:
        # Check if the highest quality version exists
        highest_q = max(qualities)
        highest_q_dir = os.path.join(out_dir, directory, str(highest_q))
        highest_q_file = os.path.join(highest_q_dir, filename[:-4] + ".jpg")
        
        if os.path.exists(highest_q_file):
            results['skipped'].extend(qualities)
            return results
        
        with rawpy.imread(img_path) as raw:
            rgb = raw.postprocess()

        img = Image.fromarray(rgb)

        for q in qualities:
            q_dir = os.path.join(out_dir, directory, str(q))
            os.makedirs(q_dir, exist_ok=True)

            out_file = os.path.join(q_dir, filename[:-4] + ".jpg")
            img.save(out_file, "JPEG", quality=q)
            results['processed'].append(q)
                
    except Exception as e:
        error_msg = f"Error processing {filename}: {e}"
        results['errors'].append(error_msg)
        
    return results

def process_img(img_path, out_dir, qualities):
    """Routes image processing to the appropriate function based on file type (DNG or RW2)."""
    filename = os.path.basename(img_path)
    ext = filename.lower().split('.')[-1]
    
    if ext == 'dng':
        return process_dng(img_path, out_dir, qualities)
    elif ext == 'rw2':
        return process_rw2(img_path, out_dir, qualities)
    else:
        return {
            'processed': [],
            'skipped': [],
            'errors': [f"Unsupported file type: {ext}"]
        }

def compress_imgs():
    """Compresses all DNG and RW2 images sequentially.
    Returns processing statistics and results."""
    if not check_drive():
        return {'error': 'External drive not found'}

    results = {
        'processed_cams': [],
        'skipped_cams': [],
        'errors': [],
        'total_processed': 0,
        'total_skipped': 0,
        'total': 0
    }
    
    orig_path = os.path.join(DRIVE, ORIG_DIR)
    comp_path = os.path.join(DRIVE, COMP_DIR)
    
    os.makedirs(comp_path, exist_ok=True)
    
    # Get all camera directories
    cams = [d for d in os.listdir(orig_path)
            if os.path.isdir(os.path.join(orig_path, d))]
    
    # Process each camera
    with tqdm(total=len(cams), desc="Processing cameras", unit="camera") as cam_pbar:
        for cam in cams:
            cam_path = os.path.join(orig_path, cam)
            img_files = [f for f in os.listdir(cam_path) if f.lower().endswith(('.dng', '.rw2'))]
            
            if not img_files:
                cam_pbar.update(1)
                continue
                
            results['total'] += len(img_files)
            
            # Process each image in the camera directory
            with tqdm(total=len(img_files), desc=f"Processing {cam}", unit="image", leave=False) as img_pbar:
                for img_file in img_files:
                    img_path = os.path.join(orig_path, cam, img_file)
                    
                    # Process the image
                    img_result = process_img(img_path, comp_path, QUALITIES)
                    
                    if img_result['processed']:
                        results['total_processed'] += 1
                        if cam not in results['processed_cams']:
                            results['processed_cams'].append(cam)
                    elif img_result['skipped']:
                        results['total_skipped'] += 1
                        if cam not in results['skipped_cams']:
                            results['skipped_cams'].append(cam)
                    elif img_result['errors']:
                        results['errors'].extend(img_result['errors'])
                    
                    img_pbar.update(1)
            
            cam_pbar.update(1)
    
    return results
