import pywt
import numpy as np
import cv2
import os


def save_spn_as_image(image_path, output_path, wavelet='db1', level=1, camera_model=None, format='png'):
    """
    Extract SPN from an image and save it as a PNG file.
    Uses wavelet decomposition to extract the sensor pattern noise.
    
    Args:
        image_path (str): Path to the input image
        output_path (str): Directory to save the SPN image
        wavelet (str, optional): Wavelet type to use. Defaults to 'db1'.
        level (int, optional): Wavelet decomposition level. Defaults to 1.
        camera_model (str, optional): Camera model name. Defaults to None.
        format (str, optional): Output image format. Defaults to 'png'.
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract SPN
    spn = extract_spn_wavelet(img, wavelet, level)
    
    # Normalize to 0-255 range
    spn_normalized = cv2.normalize(spn, None, 0, 255, cv2.NORM_MINMAX)
    spn_normalized = np.uint8(spn_normalized)
    
    # Generate output filename
    if camera_model:
        output_filename = f"{camera_model}_SPN.{format}"
    else:
        output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_SPN.{format}"
    
    # Save the SPN
    output_filepath = os.path.join(output_path, output_filename)
    cv2.imwrite(output_filepath, spn_normalized)


def extract_spn_wavelet(image, wavelet='db1', level=1):
    """
    Extract SPN using wavelet decomposition.
    Removes low-frequency content and normalizes the result.

    Args:
        image (numpy.ndarray): Input image (grayscale or color).
        wavelet (str): Wavelet type (e.g., 'db1', 'haar').
        level (int): Decomposition level.

    Returns:
        numpy.ndarray: Extracted SPN, normalized to zero mean and unit variance.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # Set approximation coefficients to zero
    coeffs = [coeffs[0] * 0] + list(coeffs[1:])  # Remove low-frequency content

    # Reconstruct residual (SPN estimate)
    spn = pywt.waverec2(coeffs, wavelet)
    spn = (spn - np.mean(spn)) / np.std(spn)  # Normalize

    return spn
