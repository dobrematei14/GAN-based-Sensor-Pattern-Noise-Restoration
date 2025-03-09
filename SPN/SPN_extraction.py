import cv2
import numpy as np
import rawpy
import bm3d
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation


# Load and process a DNG file
def load_dng(image_path):
    with rawpy.imread(image_path) as raw:
        rgb_image = raw.postprocess()
    return rgb_image  # RGB NumPy array


# Traditional SPN Extraction using BM3D
def extract_spn_bm3d(image, sigma=25):
    """ Extracts SPN from an image using BM3D denoising. """
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    denoised = bm3d.bm3d(image_gray, sigma_psd=sigma / 255.0)  # Apply BM3D
    spn = image_gray - denoised  # Residual noise pattern
    return spn


# Define a simple CNN-based SPN extractor
def build_spn_cnn(input_shape=(256, 256, 1)):
    """ Builds a CNN-based denoiser for SPN extraction. """
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(1, (3, 3), padding='same', activation='tanh')  # Output SPN residual
    ])
    return model


# Extract SPN using CNN
def extract_spn_cnn(image, model):
    """ Extract SPN using a trained CNN model. """
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_gray = cv2.resize(image_gray, (256, 256)) / 255.0  # Normalize
    image_input = np.expand_dims(image_gray, axis=(0, -1))  # Reshape for CNN
    spn = model.predict(image_input)[0, :, :, 0]  # Extracted SPN
    return spn


# Main function to extract SPN from an image
def extract_spn(image_path, method='bm3d', cnn_model=None):
    """ Extracts SPN using the chosen method (BM3D or CNN). """
    image = load_dng(image_path) if image_path.lower().endswith('.dng') else cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert OpenCV format
    if method == 'bm3d':
        return extract_spn_bm3d(image)
    elif method == 'cnn' and cnn_model is not None:
        return extract_spn_cnn(image, cnn_model)
    else:
        raise ValueError("Invalid method or missing CNN model for SPN extraction.")
