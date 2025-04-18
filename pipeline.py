import os
from dotenv import load_dotenv
from Dataset.image_compression import compress_images_with_progress
from SPN.SPN_extraction import extract_all_spns

# Load environment variables
load_dotenv()

def run_pipeline():
    """
    Run the complete image processing pipeline:
    1. Image compression
    2. SPN extraction
    
    The pipeline processes images in the following structure:
    EXTERNAL_DRIVE/
    └── Images/
        ├── Original/           # Original RAW files (DNG and RW2)
        ├── Compressed/         # Compressed JPEG files (30%, 60%, 90% quality)
        └── SPN/                # Sensor Pattern Noise files
            ├── Camera_Model/
            │   ├── original/   # Camera SPN from one RAW image
            │   └── compressed/ # Individual SPNs from compressed images
            │       ├── 30/
            │       ├── 60/
            │       └── 90/
            └── ...
    """
    print("\nStarting Image Processing Pipeline\n")
    print("=" * 50)
    
    print("\nStep 1: Image Compression")
    print("-" * 30)
    compression_results = compress_images_with_progress()
    
    if compression_results['errors']:
        print("\nCompression Errors:")
        for error in compression_results['errors']:
            print(f"- {error}")
    
    print(f"\nCompression Summary:")
    print(f"- Total images: {compression_results['total_images']}")
    print(f"- Newly processed: {compression_results['total_images_processed']}")
    print(f"- Already processed: {compression_results['total_images_skipped']}")
    print(f"- Cameras processed: {len(compression_results['processed_cameras'])}")
    print(f"- Cameras skipped: {len(compression_results['skipped_cameras'])}")
    
    print("\nStep 2: SPN Extraction")
    print("-" * 30)
    spn_results = extract_all_spns()
    
    if spn_results['original_processing']['errors']:
        print("\nSPN Extraction Errors (Original):")
        for error in spn_results['original_processing']['errors']:
            print(f"- {error}")
    
    if spn_results['compressed_processing']['errors']:
        print("\nSPN Extraction Errors (Compressed):")
        for error in spn_results['compressed_processing']['errors']:
            print(f"- {error}")
    
    print(f"\nSPN Extraction Summary:")
    print(f"- Total images processed: {spn_results['total_images_processed']}")

if __name__ == "__main__":
    run_pipeline() 