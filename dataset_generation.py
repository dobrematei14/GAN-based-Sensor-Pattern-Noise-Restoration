import torch
import numpy as np
import cv2
import os
import glob
import rawpy

from SPN.SPN_extraction import extract_spn

# Path to the external SSD drive
external_drive = "F:/"  # Same as in image_compression.py

# Base directories (update to use external SSD)
ORIGINAL_DIR = os.path.join(external_drive, 'Images/Original/')
COMPRESSED_DIR = os.path.join(external_drive, 'Images/Compressed/')


# Load trained SPN extraction CNN model
spn_model_path = 'C:\\Users\dobre\Desktop\Thesis\GAN-based-Sensor-Pattern-Noise-Restoration\spn_cnn_model.pth'
SPN_MODEL_PATH = spn_model_path
if not os.path.exists(spn_model_path):
    print(f"Warning: SPN model file {spn_model_path} not found.")
    print("Checking if an alternative model file exists...")

    # Look for alternative model files in the current directory
    potential_models = [f for f in os.listdir('.') if f.endswith('.pth')]
    if potential_models:
        print(f"Found potential model files: {', '.join(potential_models)}")
        print(f"Please update the SPN_MODEL_PATH in dataset_generation.py to use one of these files.")
    else:
        print("No model files found. Please ensure you have the correct SPN model file.")

    print("\nExiting program. Please fix the model path issue and try again.")
    exit(1)



def load_dng(image_path):
    """ Load a DNG file and convert it to an RGB NumPy array """
    with rawpy.imread(image_path) as raw:
        rgb_image = raw.postprocess()
    return rgb_image  # RGB NumPy array


def load_pair(compressed_path):
    """ Load a compressed JPEG and its corresponding original DNG image """

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
    orig_spn = extract_spn(original_path, method='cnn', cnn_model=SPN_MODEL_PATH)
    comp_spn = extract_spn(compressed_path, method='cnn', cnn_model=SPN_MODEL_PATH)

    # Normalize images to [-1,1]
    comp_img = (comp_img.astype(np.float32) / 127.5) - 1.0
    orig_img = (orig_img.astype(np.float32) / 127.5) - 1.0
    orig_spn = orig_spn.astype(np.float32) / 127.5
    comp_spn = comp_spn.astype(np.float32) / 127.5

    # One-hot encode compression level (30, 60, 90)
    compression_levels = ["30", "60", "90"]
    compression_one_hot = np.zeros(len(compression_levels))
    compression_one_hot[compression_levels.index(compression_level)] = 1.0

    # Convert to Torch tensors
    return (
        torch.tensor(comp_img, dtype=torch.float32),
        torch.tensor(orig_img, dtype=torch.float32),
        torch.tensor(comp_spn, dtype=torch.float32),
        torch.tensor(orig_spn, dtype=torch.float32),
        torch.tensor(compression_one_hot, dtype=torch.float32),
        camera_label
    )


def get_dataset(batch_size=8, shuffle_buffer=1000):
    """ Create a PyTorch dataset for training """
    # Check if external drive is available
    if not os.path.exists(external_drive):
        raise FileNotFoundError(f"External drive {external_drive} not found. Please connect the drive and try again.")

    compressed_files = glob.glob(os.path.join(COMPRESSED_DIR, '*/*/*.jpg'))  # camera/compression_level/filename.jpg

    if not compressed_files:
        raise ValueError(f"No compressed files found at {COMPRESSED_DIR}. Check if images have been processed.")

    # Create a custom dataset
    class SPNDataset(torch.utils.data.Dataset):
        def __init__(self, file_paths):
            self.file_paths = file_paths

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            return load_pair(self.file_paths[idx])

    # Create dataset and dataloader
    dataset = SPNDataset(compressed_files)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return dataloader


