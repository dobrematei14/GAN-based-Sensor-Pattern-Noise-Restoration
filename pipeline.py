import os
from dotenv import load_dotenv
from Dataset.image_compression import compress_images
from SPN.SPN_extraction import extract_all_spns

# Load environment variables
load_dotenv()

def run_pipeline():
    """
    Run the complete pipeline:
    1. Compress original DNG images to JPEG at different quality levels
    2. Extract SPNs from both original and compressed images
    
    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*50)
    print("Starting Image Processing Pipeline")
    print("="*50 + "\n")
    
    # Step 1: Image Compression
    print("\nStep 1: Image Compression")
    print("-"*30)
    compression_results = compress_images()
    
    if 'error' in compression_results:
        print(f"\n❌ Error in compression step: {compression_results['error']}")
        print("Pipeline stopped due to compression error.")
        return False
    
    print(f"\n✅ Compression completed successfully:")
    print(f"- Processed cameras: {len(compression_results['processed_cameras'])}")
    print(f"- Skipped cameras: {len(compression_results['skipped_cameras'])}")
    print(f"- Total images processed: {compression_results['total_images_processed']}")
    if compression_results['errors']:
        print(f"- Errors encountered: {len(compression_results['errors'])}")
    
    # Step 2: SPN Extraction
    print("\nStep 2: SPN Extraction")
    print("-"*30)
    spn_results = extract_all_spns()
    
    if 'error' in spn_results:
        print(f"\n❌ Error in SPN extraction step: {spn_results['error']}")
        print("Pipeline stopped due to SPN extraction error.")
        return False
    
    print(f"\n✅ SPN extraction completed successfully:")
    print(f"- Original images processed: {spn_results['original_processing']['total_images_processed']}")
    print(f"- Compressed images processed: {spn_results['compressed_processing']['total_images_processed']}")
    print(f"- Total images processed: {spn_results['total_images_processed']}")
    
    print("\n" + "="*50)
    print("✅ Pipeline completed successfully!")
    print("="*50 + "\n")
    return True

if __name__ == "__main__":
    success = run_pipeline()
    if not success:
        exit(1)  # Exit with error code if pipeline failed 