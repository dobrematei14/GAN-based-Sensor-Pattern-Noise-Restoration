import pywt
import numpy as np
import cv2
from scipy.signal import wiener
import os


def extract_camera_spn(image_path, wavelet='db1', level=1, camera_model=None):
    """
    Extract SPN from an image and generate a filename based on the camera model.

    Args:
        image_path (str): Path to the input image.
        wavelet (str): Wavelet type for decomposition (default: 'db1').
        level (int): Decomposition level for wavelet transform (default: 1).
        camera_model (str, optional): Name of the camera model (e.g., 'Canon_EOS_5D').

    Returns:
        tuple: (spn, filename)
            - spn (numpy.ndarray): Extracted SPN (2D array).
            - filename (str): Generated filename based on camera model and timestamp.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load image from {image_path}")

    # Wavelet decomposition to extract SPN
    spn = extract_spn_wavelet(image, wavelet=wavelet, level=level)
    # Generate filename
    if camera_model:
        filename = f"{camera_model}_SPN.npy"  # Include camera model in filename
    else:
        filename = f"SPN.npy"  # Default filename if no camera model is provided

    return spn, filename


def save_spn_as_image(image_path, output_path, wavelet='db1', level=1, camera_model=None, format='png'):
    """
    Extract and save SPN as a lossless image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Directory where the image will be saved.
        wavelet (str): Wavelet type for decomposition (default: 'db1').
        level (int): Decomposition level for wavelet transform (default: 1).
        camera_model (str, optional): Name of the camera model (e.g., 'Canon_EOS_5D').
        format (str): Image format ('png', 'tiff', 'bmp').
    """
    # Extract SPN and get the filename
    spn, filename = extract_camera_spn(image_path, wavelet=wavelet, level=level, camera_model=camera_model)

    # Normalize SPN to [0, 255] for 8-bit image formats
    spn_normalized = cv2.normalize(spn, None, 0, 255, cv2.NORM_MINMAX)
    spn_normalized = np.uint8(spn_normalized)  # Convert to 8-bit

    # Create the full file path
    full_path = os.path.join(output_path, f"{filename[:-4]}.{format}")

    # Save as lossless image
    if format == 'png':
        cv2.imwrite(full_path, spn_normalized, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # PNG with no compression
    elif format == 'tiff':
        cv2.imwrite(full_path, spn_normalized)  # TIFF (lossless by default)
    elif format == 'bmp':
        cv2.imwrite(full_path, spn_normalized)  # BMP (uncompressed)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'png', 'tiff', or 'bmp'.")

    print(f"SPN saved successfully at: {full_path}")


def extract_spn_wavelet(image, wavelet='db1', level=1):  # current preferred method
    """
    Extract SPN using wavelet decomposition.

    Args:
        image (numpy.ndarray): Input image (grayscale or color).
        wavelet (str): Wavelet type (e.g., 'db1', 'haar').
        level (int): Decomposition level.

    Returns:
        numpy.ndarray: Extracted SPN.
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


def extract_spn_threshold(image, wavelet='db4', level=3, threshold=0.1):
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # Threshold detail coefficients
    thresholded_coeffs = [coeffs[0]] + [
        tuple(pywt.threshold(c, threshold * np.max(c), mode='soft') for c in level_coeffs)
        for level_coeffs in coeffs[1:]
    ]

    # Reconstruct denoised image
    denoised = pywt.waverec2(thresholded_coeffs, wavelet)

    # Residual = Original - Denoised (SPN estimate)
    spn = image - denoised
    spn = (spn - np.mean(spn)) / np.std(spn)

    return spn


def extract_spn_wiener_wavelet(image, wavelet='sym4'):
    # Wiener filtering
    denoised = wiener(image, mysize=(5, 5))

    # Residual after Wiener filter
    residual = image - denoised

    # Further refine with wavelet decomposition
    coeffs = pywt.wavedec2(residual, wavelet, level=1)
    coeffs = [coeffs[0] * 0] + list(coeffs[1:])  # Remove low-frequency artifacts
    spn = pywt.waverec2(coeffs, wavelet)

    return spn


def extract_spn_multiscale(image, wavelet='db2', levels=4):
    spn_estimates = []
    for level in range(1, levels + 1):
        coeffs = pywt.wavedec2(image, wavelet, level=level)
        coeffs = [coeffs[0] * 0] + list(coeffs[1:])
        spn = pywt.waverec2(coeffs, wavelet)
        spn_estimates.append(spn)

    # Average across scales
    spn_avg = np.mean(spn_estimates, axis=0)
    spn_avg = (spn_avg - np.mean(spn_avg)) / np.std(spn_avg)

    return spn_avg
