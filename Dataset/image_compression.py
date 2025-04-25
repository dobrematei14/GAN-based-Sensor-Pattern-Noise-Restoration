import os
import rawpy
import imageio
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from functools import partial

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
        with rawpy.imread(img_path) as raw:
            rgb = raw.postprocess()
            img = Image.fromarray(rgb)
            
            for q in qualities:
                q_dir = os.path.join(out_path, str(q))
                os.makedirs(q_dir, exist_ok=True)
                
                out_file = os.path.join(q_dir, f"{os.path.splitext(filename)[0]}.jpg")
                
                if os.path.exists(out_file):
                    results['skipped'].append(q)
                    continue
                
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
        with rawpy.imread(img_path) as raw:
            rgb = raw.postprocess()

        img = Image.fromarray(rgb)

        for q in qualities:
            q_dir = os.path.join(out_dir, directory, str(q))
            os.makedirs(q_dir, exist_ok=True)

            out_file = os.path.join(q_dir, filename[:-4] + ".jpg")

            if not os.path.exists(out_file):
                img.save(out_file, "JPEG", quality=q)
                results['processed'].append(q)
            else:
                results['skipped'].append(q)
                
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

def process_single(args):
    """Processes a single image with its parameters for parallel processing."""
    img_file, cam, orig_path, comp_path = args
    img_path = os.path.join(orig_path, cam, img_file)
    
    result = {
        'filename': img_file,
        'processed': False,
        'skipped': False,
        'error': None
    }
    
    try:
        comp_results = process_img(img_path, comp_path, QUALITIES)
        
        if comp_results['processed']:
            result['processed'] = True
        elif comp_results['errors']:
            result['error'] = comp_results['errors'][0]
        else:
            result['skipped'] = True
            
    except Exception as e:
        result['error'] = str(e)
    
    return result

def compress_imgs():
    """Compresses all DNG and RW2 images in parallel using all available CPU cores.
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
    
    cams = [d for d in os.listdir(orig_path)
            if os.path.isdir(os.path.join(orig_path, d))]
    
    for cam in cams:
        cam_path = os.path.join(orig_path, cam)
        img_files = [f for f in os.listdir(cam_path) if f.lower().endswith(('.dng', '.rw2'))]
        
        if not img_files:
            continue
            
        results['total'] += len(img_files)
        
        with Pool() as pool:
            process_func = partial(process_single, 
                                 cam=cam,
                                 orig_path=orig_path,
                                 comp_path=comp_path)
            
            for result in pool.imap(process_func, img_files):
                if result['processed']:
                    results['total_processed'] += 1
                elif result['skipped']:
                    results['total_skipped'] += 1
                elif result['error']:
                    results['errors'].append(result['error'])
    
    return results
