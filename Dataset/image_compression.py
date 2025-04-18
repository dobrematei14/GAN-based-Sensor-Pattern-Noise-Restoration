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
EXTERNAL_DRIVE = os.getenv('EXTERNAL_DRIVE')
ORIGINAL_IMAGES_DIR = os.getenv('ORIGINAL_IMAGES_DIR')
COMPRESSED_IMAGES_DIR = os.getenv('COMPRESSED_IMAGES_DIR')

# Parse quality levels from environment variable
QUALITY_LEVELS = [int(q) for q in os.getenv('QUALITY_LEVELS').split(',')]

def check_external_drive():
    """
    Check if the external drive is connected and accessible.
    
    Returns:
        bool: True if drive is accessible, False otherwise
    """
    if not os.path.exists(EXTERNAL_DRIVE):
        print(f"Error: External drive {EXTERNAL_DRIVE} not found. Please connect the drive and try again.")
        return False
    return True

def process_dng_image(input_path, output_base_dir, quality_levels):
    """
    Process a single DNG (Digital Negative) image and create compressed JPEG versions at different quality levels.
    
    Args:
        input_path (str): Path to the input DNG file
        output_base_dir (str): Base directory for compressed images
        quality_levels (list): List of quality levels (1-100) to compress to
        
    Returns:
        dict: Processing results containing:
            - processed_qualities: List of quality levels successfully processed
            - skipped_qualities: List of quality levels skipped (already exist)
            - errors: List of any errors encountered during processing
    """
    results = {
        'processed_qualities': [],
        'skipped_qualities': [],
        'errors': []
    }
    
    filename = os.path.basename(input_path)
    directory = os.path.basename(os.path.dirname(input_path))
    
    # Create output directory for this camera model
    output_dir = os.path.join(output_base_dir, directory)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Open the DNG file
        with rawpy.imread(input_path) as raw:
            # Convert to RGB
            rgb = raw.postprocess()
            # Convert to PIL Image
            img = Image.fromarray(rgb)
            
            # Process each quality level
            for quality in quality_levels:
                # Create quality subdirectory
                quality_dir = os.path.join(output_dir, str(quality))
                os.makedirs(quality_dir, exist_ok=True)
                
                # Set output path
                output_path = os.path.join(quality_dir, f"{os.path.splitext(filename)[0]}.jpg")
                
                # Skip if file already exists
                if os.path.exists(output_path):
                    results['skipped_qualities'].append(quality)
                    continue
                
                # Save as JPEG with specified quality
                img.save(output_path, 'JPEG', quality=quality)
                results['processed_qualities'].append(quality)
                
    except Exception as e:
        results['errors'].append(f"Error processing {input_path}: {str(e)}")
    
    return results

def process_rw2_image(input_path, output_base_dir, quality_levels):
    """
    Process a single RW2 (Panasonic RAW) image and create compressed JPEG versions at different quality levels.
    
    Args:
        input_path (str): Path to the input RW2 file
        output_base_dir (str): Base directory for compressed images
        quality_levels (list): List of quality levels (1-100) to compress to
        
    Returns:
        dict: Processing results containing:
            - processed_qualities: List of quality levels successfully processed
            - skipped_qualities: List of quality levels skipped (already exist)
            - errors: List of any errors encountered during processing
    """
    results = {
        'processed_qualities': [],
        'skipped_qualities': [],
        'errors': []
    }
    
    filename = os.path.basename(input_path)
    directory = os.path.basename(os.path.dirname(input_path))
    
    try:
        # Convert RW2 to RGB using rawpy
        with rawpy.imread(input_path) as raw:
            rgb = raw.postprocess()

        # Create PIL Image for compression
        image = Image.fromarray(rgb)

        # Compress at different quality levels
        for quality in quality_levels:
            # Create quality-specific directory
            quality_dir = os.path.join(output_base_dir, directory, str(quality))
            os.makedirs(quality_dir, exist_ok=True)

            # Save compressed image
            output_path = os.path.join(quality_dir, filename[:-4] + ".jpg")

            # Check if this specific quality version already exists
            if not os.path.exists(output_path):
                image.save(output_path, "JPEG", quality=quality)
                results['processed_qualities'].append(quality)
            else:
                results['skipped_qualities'].append(quality)
                
    except Exception as e:
        error_msg = f"Error processing {filename}: {e}"
        results['errors'].append(error_msg)
        
    return results

def process_image(input_path, output_base_dir, quality_levels):
    """
    Process a single raw image file (DNG or RW2) and create compressed JPEG versions.
    Routes to the appropriate processing function based on file type.
    
    Args:
        input_path (str): Path to the input raw image file
        output_base_dir (str): Base directory for compressed images
        quality_levels (list): List of quality levels (1-100) to compress to
        
    Returns:
        dict: Processing results containing:
            - processed_qualities: List of quality levels successfully processed
            - skipped_qualities: List of quality levels skipped (already exist)
            - errors: List of any errors encountered during processing
    """
    filename = os.path.basename(input_path)
    file_extension = filename.lower().split('.')[-1]
    
    if file_extension == 'dng':
        return process_dng_image(input_path, output_base_dir, quality_levels)
    elif file_extension == 'rw2':
        return process_rw2_image(input_path, output_base_dir, quality_levels)
    else:
        return {
            'processed_qualities': [],
            'skipped_qualities': [],
            'errors': [f"Unsupported file type: {file_extension}"]
        }

def process_single_image(args):
    """
    Process a single image with its parameters.
    This function is designed to be used with multiprocessing.
    
    Args:
        args (tuple): (raw_file, camera_model, original_base_path, compressed_base_path)
        
    Returns:
        dict: Processing results
    """
    raw_file, camera_model, original_base_path, compressed_base_path = args
    raw_path = os.path.join(original_base_path, camera_model, raw_file)
    
    result = {
        'filename': raw_file,
        'processed': False,
        'skipped': False,
        'error': None
    }
    
    try:
        # Process the image
        compression_results = process_image(raw_path, compressed_base_path, QUALITY_LEVELS)
        
        if compression_results['processed_qualities']:
            result['processed'] = True
        elif compression_results['errors']:
            result['error'] = compression_results['errors'][0]
        else:
            result['skipped'] = True
            
    except Exception as e:
        result['error'] = str(e)
    
    return result

def compress_images_with_progress():
    """
    Compress all DNG and RW2 images in parallel using all available CPU cores.
    Shows progress bars for camera models and individual image processing.
    
    Returns:
        dict: Dictionary containing processing results and statistics:
            - processed_cameras: List of successfully processed camera directories
            - skipped_cameras: List of skipped camera directories
            - errors: List of any errors encountered
            - total_images_processed: Total count of newly processed images
            - total_images_skipped: Total count of skipped images
            - total_images: Total count of all images
    """
    # Check if external drive is accessible
    if not check_external_drive():
        return {'error': 'External drive not found'}

    results = {
        'processed_cameras': [],
        'skipped_cameras': [],
        'errors': [],
        'total_images_processed': 0,
        'total_images_skipped': 0,
        'total_images': 0
    }
    
    # Get the base paths
    original_base_path = os.path.join(EXTERNAL_DRIVE, ORIGINAL_IMAGES_DIR)
    compressed_base_path = os.path.join(EXTERNAL_DRIVE, COMPRESSED_IMAGES_DIR)
    
    # Create compressed images directory if it doesn't exist
    os.makedirs(compressed_base_path, exist_ok=True)
    
    # Get list of camera models
    camera_models = [d for d in os.listdir(original_base_path)
                    if os.path.isdir(os.path.join(original_base_path, d))]
    
    # Process each camera model
    for camera_model in tqdm(camera_models, desc="Processing camera models", unit="camera"):
        camera_dir = os.path.join(original_base_path, camera_model)
        
        # Get all RAW files
        raw_files = [f for f in os.listdir(camera_dir)
                    if f.lower().endswith(('.dng', '.rw2'))]
        
        if not raw_files:
            continue
            
        # Pre-check which files need processing
        files_to_process = []
        for raw_file in raw_files:
            needs_processing = False
            for quality in QUALITY_LEVELS:
                quality_dir = os.path.join(compressed_base_path, camera_model, str(quality))
                output_path = os.path.join(quality_dir, f"{os.path.splitext(raw_file)[0]}.jpg")
                if not os.path.exists(output_path):
                    needs_processing = True
                    break
            if needs_processing:
                files_to_process.append(raw_file)
            else:
                results['total_images_skipped'] += 1
                results['total_images'] += 1
        
        if not files_to_process:
            results['skipped_cameras'].append(camera_model)
            continue
            
        # Prepare arguments for parallel processing
        process_args = [(f, camera_model, original_base_path, compressed_base_path) 
                       for f in files_to_process]
        
        # Create a process pool using all available cores
        num_cores = multiprocessing.cpu_count()
        with Pool(num_cores) as pool:
            # Process images in parallel with progress bar
            for result in tqdm(
                pool.imap_unordered(process_single_image, process_args),
                total=len(files_to_process),
                desc=f"Processing {camera_model}",
                unit="image",
                leave=False
            ):
                if result['processed']:
                    results['total_images_processed'] += 1
                elif result['skipped']:
                    results['total_images_skipped'] += 1
                if result['error']:
                    results['errors'].append(result['error'])
                results['total_images'] += 1
        
        if results['total_images_processed'] > 0:
            results['processed_cameras'].append(camera_model)
        else:
            results['skipped_cameras'].append(camera_model)
    
    return results
