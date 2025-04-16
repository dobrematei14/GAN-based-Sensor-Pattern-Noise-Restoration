import os
import rawpy
import imageio
from PIL import Image
from dotenv import load_dotenv

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

def process_single_image(input_path, output_base_dir, quality_levels):
    """
    Process a single DNG image and create compressed versions at different quality levels.
    
    Args:
        input_path (str): Path to the input DNG file
        output_base_dir (str): Base directory for compressed images
        quality_levels (list): List of quality levels to compress to
        
    Returns:
        dict: Processing results for this image
    """
    results = {
        'processed_qualities': [],
        'skipped_qualities': [],
        'errors': []
    }
    
    filename = os.path.basename(input_path)
    directory = os.path.basename(os.path.dirname(input_path))
    
    try:
        # Convert DNG to RGB using rawpy
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
                print(f"Saved {output_path}")
                results['processed_qualities'].append(quality)
            else:
                print(f"Skipping existing file: {output_path}")
                results['skipped_qualities'].append(quality)
                
    except Exception as e:
        error_msg = f"Error processing {filename}: {e}"
        print(error_msg)
        results['errors'].append(error_msg)
        
    return results

def process_camera_directory(dir_path, output_base_dir, quality_levels):
    """
    Process all DNG files in a camera directory.
    
    Args:
        dir_path (str): Path to the camera directory
        output_base_dir (str): Base directory for compressed images
        quality_levels (list): List of quality levels to compress to
        
    Returns:
        dict: Processing results for this camera directory
    """
    results = {
        'processed_images': [],
        'skipped_images': [],
        'errors': [],
        'total_images_processed': 0
    }
    
    directory = os.path.basename(dir_path)
    print(f"Processing directory: {directory}")

    # Process each DNG file in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith(".DNG"):
            input_path = os.path.join(dir_path, filename)

            # Check if this image has already been processed for all quality levels
            already_processed = True
            for quality in quality_levels:
                quality_dir = os.path.join(output_base_dir, directory, str(quality))
                output_path = os.path.join(quality_dir, filename[:-4] + ".jpg")
                if not os.path.exists(output_path):
                    already_processed = False
                    break

            # Skip if the image has already been processed for all quality levels
            if already_processed:
                print(f"Skipping already processed image: {filename}")
                results['skipped_images'].append(filename)
                continue

            # Process the image
            image_results = process_single_image(input_path, output_base_dir, quality_levels)
            
            if not image_results['errors']:
                results['processed_images'].append(filename)
                results['total_images_processed'] += 1
            else:
                results['errors'].extend(image_results['errors'])
                
    return results

def compress_images(input_base_dir=None, output_base_dir=None, quality_levels=None):
    """
    Main function to compress images from DNG to JPEG at different quality levels.
    
    Args:
        input_base_dir (str, optional): Base directory containing original DNG images
        output_base_dir (str, optional): Base directory for compressed images
        quality_levels (list, optional): List of quality levels to compress to
        
    Returns:
        dict: Processing results for all images
    """
    # Use environment variables if not provided
    input_base_dir = input_base_dir or os.path.join(EXTERNAL_DRIVE, ORIGINAL_IMAGES_DIR)
    output_base_dir = output_base_dir or os.path.join(EXTERNAL_DRIVE, COMPRESSED_IMAGES_DIR)
    quality_levels = quality_levels or QUALITY_LEVELS
    
    # Check if external drive is accessible
    if not check_external_drive():
        return {'error': 'External drive not found'}
    
    results = {
        'processed_cameras': [],
        'skipped_cameras': [],
        'errors': [],
        'total_images_processed': 0
    }
    
    # Process each camera directory
    for directory in os.listdir(input_base_dir):
        dir_path = os.path.join(input_base_dir, directory)
        
        # Skip if not a directory
        if not os.path.isdir(dir_path):
            continue
            
        # Process the camera directory
        camera_results = process_camera_directory(dir_path, output_base_dir, quality_levels)
        
        if camera_results['total_images_processed'] > 0:
            results['processed_cameras'].append(directory)
            results['total_images_processed'] += camera_results['total_images_processed']
        else:
            results['skipped_cameras'].append(directory)
            
        results['errors'].extend(camera_results['errors'])
        
    return results
