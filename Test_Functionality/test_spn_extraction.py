import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import rawpy

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from SPN.SPN_extraction_methods import extract_spn_wavelet, extract_spn_denoise_wavelet

def visualize_results(original, spn_wavelet, spn_denoise, title):
    """
    Visualize the original image and both SPN extraction results.
    Includes both raw SPN values and thresholded hot pixel visualization.
    
    Args:
        original (numpy.ndarray): Original image
        spn_wavelet (numpy.ndarray): SPN extracted using wavelet decomposition
        spn_denoise (numpy.ndarray): SPN extracted using denoise_wavelet
        title (str): Title for the plot
    """
    # Create figure with custom gridspec
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1])
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Raw SPN values
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(spn_wavelet, cmap='gray')
    ax2.set_title('SPN Wavelet (Raw)')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(spn_denoise, cmap='gray')
    ax3.set_title('SPN Denoise (Raw)')
    ax3.axis('off')
    
    # Calculate thresholds (e.g., 2 standard deviations from mean)
    threshold_wavelet = np.mean(spn_wavelet) + 2 * np.std(spn_wavelet)
    threshold_denoise = np.mean(spn_denoise) + 2 * np.std(spn_denoise)
    
    # Create binary-like visualizations (hot pixels are black)
    hot_pixels_wavelet_black = 1 - (spn_wavelet > threshold_wavelet).astype(float)
    hot_pixels_denoise_black = 1 - (spn_denoise > threshold_denoise).astype(float)
    
    # Create binary-like visualizations (hot pixels are white)
    hot_pixels_wavelet_white = (spn_wavelet > threshold_wavelet).astype(float)
    hot_pixels_denoise_white = (spn_denoise > threshold_denoise).astype(float)
    
    # Add histograms of SPN values with statistics
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(spn_wavelet.ravel(), bins=50, alpha=0.5, label='Wavelet', density=True)
    ax4.hist(spn_denoise.ravel(), bins=50, alpha=0.5, label='Denoise', density=True)
    ax4.axvline(x=threshold_wavelet, color='blue', linestyle='--', alpha=0.5)
    ax4.axvline(x=threshold_denoise, color='orange', linestyle='--', alpha=0.5)
    ax4.set_title('SPN Value Distribution')
    ax4.legend()
    
    # Add statistics as text box
    stats_text = (
        f"Wavelet Method:\n"
        f"Mean: {np.mean(spn_wavelet):.3f}\n"
        f"Std:  {np.std(spn_wavelet):.3f}\n"
        f"Min:  {np.min(spn_wavelet):.3f}\n"
        f"Max:  {np.max(spn_wavelet):.3f}\n"
        f"Hot Pixels: {np.sum(hot_pixels_wavelet_white):.0f} "
        f"({(np.sum(hot_pixels_wavelet_white)/hot_pixels_wavelet_white.size)*100:.3f}%)\n\n"
        f"Denoise Method:\n"
        f"Mean: {np.mean(spn_denoise):.3f}\n"
        f"Std:  {np.std(spn_denoise):.3f}\n"
        f"Min:  {np.min(spn_denoise):.3f}\n"
        f"Max:  {np.max(spn_denoise):.3f}\n"
        f"Hot Pixels: {np.sum(hot_pixels_denoise_white):.0f} "
        f"({(np.sum(hot_pixels_denoise_white)/hot_pixels_denoise_white.size)*100:.3f}%)"
    )
    
    # Add text box to the plot
    ax4.text(1.05, 1.0, stats_text,
            transform=ax4.transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict(boxstyle='round',
                     facecolor='white',
                     alpha=0.8))
    
    # Hot pixels visualization (black = hot)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(hot_pixels_wavelet_black, cmap='binary')
    ax5.set_title(f'Hot Pixels (Wavelet)\nblack > {threshold_wavelet:.3f}')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(hot_pixels_denoise_black, cmap='binary')
    ax6.set_title(f'Hot Pixels (Denoise)\nblack > {threshold_denoise:.3f}')
    ax6.axis('off')
    
    # Hot pixels visualization (white = hot)
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.imshow(hot_pixels_wavelet_white, cmap='binary')
    ax7.set_title(f'Hot Pixels (Wavelet)\nwhite > {threshold_wavelet:.3f}')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.imshow(hot_pixels_denoise_white, cmap='binary')
    ax8.set_title(f'Hot Pixels (Denoise)\nwhite > {threshold_denoise:.3f}')
    ax8.axis('off')
    
    # Add explanation text for the bottom row
    explanation_text = (
        "Hot pixels shown in white\n"
        "Cold pixels shown in black\n"
        "(Alternative visualization)"
    )
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.text(0.5, 0.5, explanation_text,
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=10,
             transform=ax9.transAxes)
    ax9.axis('off')
    
    plt.suptitle(title, y=0.95)
    plt.tight_layout()
    plt.show()

def load_image(image_path):
    """
    Load an image, handling both regular images and RAW files.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Grayscale image
    """
    if image_path.lower().endswith(('.dng', '.raw', '.cr2', '.nef')):
        # Handle RAW files
        with rawpy.imread(image_path) as raw:
            # Get the raw image data
            raw_image = raw.raw_image
            
            # Convert to grayscale by averaging the color channels
            if len(raw_image.shape) == 3:
                gray_image = np.mean(raw_image, axis=2)
            else:
                gray_image = raw_image
                
            # Normalize to 0-255 range
            gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
            gray_image = np.uint8(gray_image)
            
            return gray_image
    else:
        # Handle regular image files
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def test_spn_extraction(image_path):
    """
    Test both SPN extraction methods on a given image.
    
    Args:
        image_path (str): Path to the test image
    """
    try:
        # Load and convert to grayscale
        gray_image = load_image(image_path)
        
        # Extract SPN using both methods
        spn_wavelet = extract_spn_wavelet(gray_image, wavelet='db1', level=1)
        spn_denoise = extract_spn_denoise_wavelet(gray_image, wavelet='db8', is_color=False)
        
        # Visualize results
        visualize_results(
            gray_image,
            spn_wavelet,
            spn_denoise,
            f"SPN Extraction Results - {os.path.basename(image_path)}"
        )
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        test_image_path = "F:\\ip15pm\\IMG_1277.DNG"  # Your DNG image path
    test_spn_extraction(test_image_path) 