from Image_loading import retrieveDNG
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.restoration import denoise_wavelet


def convert_to_float(image):
    """ Convert image to float32 range [0,1] for processing. """
    return image.astype(np.float32) / 255.0


def wavelet_denoise(image):
    """ Apply stronger denoising using wavelet thresholding. """
    image = convert_to_float(image)
    denoised = denoise_wavelet(
        image,
        convert2ycbcr=True,
        method='BayesShrink',
        mode='soft',
        channel_axis=-1
    )
    return image, denoised


def apply_median_blur(image, kernel_size=5):
    """ Apply median filtering to remove large structures before SPN extraction. """
    return cv2.medianBlur(image, kernel_size)


def extract_SPN(image):
    """ Extract Sensor Pattern Noise (SPN) using wavelet denoising and median filtering. """
    # Convert to grayscale to remove color dependency
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply median filtering to suppress image details
    median_filtered = apply_median_blur(grayscale, kernel_size=5)

    # Apply wavelet denoising
    _, denoised = wavelet_denoise(median_filtered)

    # Compute SPN (original - denoised)
    spn = median_filtered - denoised

    # Normalize the SPN to enhance visibility
    spn_normalized = (spn - np.min(spn)) / (np.max(spn) - np.min(spn))

    return spn_normalized


def preview_SPN(dng_path):
    """ Extract and display the SPN from a DNG file. """
    dng = retrieveDNG(dng_path)
    spn = extract_SPN(dng)

    plt.figure(figsize=(6, 6))
    plt.imshow(spn, cmap="gray")
    plt.title("Extracted Sensor Pattern Noise (SPN)")
    plt.axis("off")
    plt.show()


def save_SPN(dng_path, save_path):
    """ Extract the SPN from a DNG file and save it as an image. """
    dng = retrieveDNG(dng_path)
    spn = extract_SPN(dng)

    # Save as PNG (grayscale)
    plt.imsave(save_path, spn, cmap="gray")

    print(f"SPN saved to {save_path}")


# Example usage:
save_SPN("Images/Original Format/1.DNG", "Images/SPN/1.png")
