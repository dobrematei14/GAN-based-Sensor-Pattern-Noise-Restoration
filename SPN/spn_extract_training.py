import tensorflow as tf
import numpy as np
import cv2
import os
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from SPN_extraction import extract_spn_bm3d
import rawpy

# Paths to dataset
IMAGE_DIR = "Images/Original/"
SAVE_MODEL_PATH = "spn_cnn_model.h5"


# Load images and compute SPN ground truth using BM3D

def load_training_data(image_dir, img_size=(256, 256)):
    image_paths = glob.glob(os.path.join(image_dir, "*/*.DNG"))
    X, Y = [], []

    for img_path in image_paths:
        # Read DNG file using rawpy
        with rawpy.imread(img_path) as raw:
            image = raw.postprocess()
            # Convert to uint8 (0-255 range)
            image = (image / image.max() * 255).astype(np.uint8)
            image = cv2.resize(image, img_size)

        # Extract ground-truth SPN using BM3D
        spn = extract_spn_bm3d(image)
        spn = cv2.resize(spn, img_size)

        # Normalize to 0-1 range after processing
        X.append(np.expand_dims(image[:, :, 0] / 255.0, axis=-1))
        Y.append(np.expand_dims(spn / 255.0, axis=-1))

    return np.array(X), np.array(Y)

# Build CNN model for SPN extraction
def build_spn_cnn(input_shape=(256, 256, 1)):
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(1, (3, 3), padding='same', activation='tanh')
    ])
    return model


# Train SPN CNN model
def train_spn_cnn():
    print("Loading training data...")
    X_train, Y_train = load_training_data(IMAGE_DIR)

    model = build_spn_cnn()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='mse', metrics=['mae'])

    print("Training SPN extraction model...")
    model.fit(X_train, Y_train, batch_size=8, epochs=50, validation_split=0.1)

    model.save(SAVE_MODEL_PATH)
    print(f"Model saved to {SAVE_MODEL_PATH}")


if __name__ == "__main__":
    train_spn_cnn()
