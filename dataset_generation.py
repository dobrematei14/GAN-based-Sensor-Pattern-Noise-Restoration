import tensorflow as tf
import rawpy
import numpy as np
import cv2
import os
import glob
from SPN.SPN_extraction import extract_spn

# Path to the external SSD drive
external_drive = "F:/"  # Same as in image_compression.py

# Base directories (update to use external SSD)
ORIGINAL_DIR = os.path.join(external_drive, 'Images/Original/')
COMPRESSED_DIR = os.path.join(external_drive, 'Images/Compressed/')

# SPN model path - assuming it's in your local project directory
SPN_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Code/spn_cnn_model.h5')

# Load trained SPN extraction CNN model
spn_cnn_model = tf.keras.models.load_model(
    SPN_MODEL_PATH,
    custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
)


def load_dng(image_path):
    """ Load a DNG file and convert it to an RGB NumPy array """
    with rawpy.imread(image_path) as raw:
        rgb_image = raw.postprocess()
    return rgb_image  # RGB NumPy array


def load_pair(compressed_path):
    """ Load a compressed JPEG and its corresponding original DNG image """
    compressed_path = compressed_path.decode("utf-8")  # Convert byte string to normal string

    # Extract camera label, compression level, and filename
    path_parts = compressed_path.split(os.sep)
    camera_label, compression_level, filename = path_parts[-3], path_parts[-2], path_parts[-1]

    # Match with the corresponding original image
    filename_dng = os.path.splitext(filename)[0] + ".DNG"
    original_path = os.path.join(ORIGINAL_DIR, camera_label, filename_dng)

    # Load images
    comp_img = cv2.imread(compressed_path)[:, :, ::-1]  # OpenCV loads BGR, convert to RGB
    orig_img = load_dng(original_path)  # Load original DNG

    # Resize images (optional, adjust size if needed)
    target_size = (256, 256)
    comp_img = cv2.resize(comp_img, target_size, interpolation=cv2.INTER_AREA)
    orig_img = cv2.resize(orig_img, target_size, interpolation=cv2.INTER_AREA)

    # Extract SPN from both original and compressed images
    orig_spn = extract_spn(original_path, method='cnn', cnn_model=spn_cnn_model)
    comp_spn = extract_spn(compressed_path, method='cnn', cnn_model=spn_cnn_model)

    # Normalize images to [-1,1]
    comp_img = (comp_img.astype(np.float32) / 127.5) - 1.0
    orig_img = (orig_img.astype(np.float32) / 127.5) - 1.0
    orig_spn = orig_spn.astype(np.float32) / 127.5
    comp_spn = comp_spn.astype(np.float32) / 127.5

    # One-hot encode compression level (30, 60, 90)
    compression_levels = ["30", "60", "90"]
    compression_one_hot = np.zeros(len(compression_levels))
    compression_one_hot[compression_levels.index(compression_level)] = 1.0

    # Convert to TensorFlow tensors
    return (
        tf.convert_to_tensor(comp_img),
        tf.convert_to_tensor(orig_img),
        tf.convert_to_tensor(comp_spn),
        tf.convert_to_tensor(orig_spn),
        tf.convert_to_tensor(compression_one_hot, dtype=tf.float32),
        tf.convert_to_tensor(camera_label)
    )


def get_dataset(batch_size=8, shuffle_buffer=1000):
    """ Create a TensorFlow dataset for training """
    # Check if external drive is available
    if not os.path.exists(external_drive):
        raise FileNotFoundError(f"External drive {external_drive} not found. Please connect the drive and try again.")

    compressed_files = glob.glob(os.path.join(COMPRESSED_DIR, '*/*/*.jpg'))  # camera/compression_level/filename.jpg

    if not compressed_files:
        print(f"Warning: No compressed files found at {COMPRESSED_DIR}. Check if images have been processed.")

    dataset = tf.data.Dataset.from_tensor_slices(compressed_files)
    dataset = dataset.map(lambda x: tf.py_function(load_pair, [x],
                                                   [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                                    tf.string]))
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
