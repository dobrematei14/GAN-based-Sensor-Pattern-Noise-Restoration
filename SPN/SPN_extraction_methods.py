import pywt
import numpy as np
import cv2
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


def extract_compressed_images_spn(compressed_base_path, output_base_path, wavelet='db1', level=1):
    """
    Extract SPNs from compressed images across different quality levels
    and save them in the specified directory structure.

    Args:
        compressed_base_path (str): Base path for compressed images (e.g., F:\\Images\\Compressed)
        output_base_path (str): Base path for saving SPNs (e.g., F:\\Images\\SPN)
        wavelet (str): Wavelet type for decomposition (default: 'db1')
        level (int): Decomposition level for wavelet transform (default: 1)
    """
    # Get list of camera models (directories in the compressed path)
    camera_models = [d for d in os.listdir(compressed_base_path)
                     if os.path.isdir(os.path.join(compressed_base_path, d))]

    print(f"Found {len(camera_models)} camera models: {', '.join(camera_models)}")

    # Process each camera model
    for camera_model in camera_models:
        camera_dir = os.path.join(compressed_base_path, camera_model)

        # Get list of quality levels (directories inside each camera model)
        quality_levels = [q for q in os.listdir(camera_dir)
                          if os.path.isdir(os.path.join(camera_dir, q))]

        print(f"Processing camera model: {camera_model}, quality levels: {quality_levels}")

        # Process each quality level
        for quality_level in quality_levels:
            quality_dir = os.path.join(camera_dir, quality_level)

            # Create output directory structure
            output_dir = os.path.join(output_base_path, camera_model, "SPN_compressed", quality_level)
            os.makedirs(output_dir, exist_ok=True)

            # Get all compressed images in this quality level
            compressed_images = [f for f in os.listdir(quality_dir)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            print(
                f"Found {len(compressed_images)} compressed images for {camera_model} at quality level {quality_level}")

            # Process each compressed image
            for image_file in compressed_images:
                image_path = os.path.join(quality_dir, image_file)
                output_filename = f"{os.path.splitext(image_file)[0]}_SPN.png"
                output_path = os.path.join(output_dir, output_filename)

                # Skip if the SPN already exists
                if os.path.exists(output_path):
                    print(f"SPN already exists for {image_file}, skipping")
                    continue

                # Extract and save the SPN
                try:
                    # Use the existing save_spn_as_image function
                    save_spn_as_image(
                        image_path=image_path,
                        output_path=output_dir,
                        wavelet=wavelet,
                        level=level,
                        camera_model=os.path.splitext(image_file)[0],
                        format='png'
                    )

                    print(f"Extracted and saved SPN for {image_file}")

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    print("SPN extraction from compressed images completed successfully!")


def extract_spn_wavelet(image, wavelet='db1', level=1):
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
