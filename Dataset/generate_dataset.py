import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get paths from environment variables
EXTERNAL_DRIVE = os.getenv('EXTERNAL_DRIVE')
COMPRESSED_IMAGES_DIR = os.getenv('COMPRESSED_IMAGES_DIR')
ORIGINAL_IMAGES_DIR = os.getenv('ORIGINAL_IMAGES_DIR')
QUALITY_LEVELS = [int(q) for q in os.getenv('QUALITY_LEVELS').split(',')]

def create_image_pairs(uncompressed_path, compressed_path):
    """Create pairs of compressed and uncompressed images with camera model info."""
    pairs = []

    for camera_model in os.listdir(uncompressed_path):
        camera_dir_path = os.path.join(uncompressed_path, camera_model)

        # Skip if not a directory
        if not os.path.isdir(camera_dir_path):
            continue

        # For each compression level
        for quality_level in QUALITY_LEVELS:
            compressed_dir = os.path.join(compressed_path, camera_model, str(quality_level))

            # Skip if compressed directory doesn't exist
            if not os.path.exists(compressed_dir):
                continue

            # Find all image pairs
            for filename in os.listdir(camera_dir_path):
                if filename.endswith(".DNG"):
                    # Original file is DNG, compressed is JPG
                    uncompressed_img_path = os.path.join(camera_dir_path, filename)
                    compressed_img_path = os.path.join(compressed_dir, filename[:-4] + ".jpg")

                    if os.path.exists(compressed_img_path):
                        # Include camera model and compression level in the pair information
                        pairs.append({
                            "compressed_path": compressed_img_path,
                            "uncompressed_path": uncompressed_img_path,
                            "camera_model": camera_model,
                            "quality_level": quality_level
                        })

    return pairs


# Used for one-hot encoding
def count_cameras(base_path=None):
    """
    Count the number of unique camera models in the dataset.

    Parameters:
    -----------
    base_path : str, optional
        Base path to the compressed images directory. If None, uses environment variable.

    Returns:
    --------
    int : The number of unique camera models
    list : List of camera model names
    """
    if base_path is None:
        base_path = os.path.join(EXTERNAL_DRIVE, COMPRESSED_IMAGES_DIR)

    if not os.path.exists(base_path):
        print(f"Path does not exist: {base_path}")
        return 0, []

    # Get all subdirectories in the base path (these are camera models)
    camera_models = [d for d in os.listdir(base_path)
                     if os.path.isdir(os.path.join(base_path, d))]

    num_cameras = len(camera_models)

    print(f"Found {num_cameras} camera models: {', '.join(camera_models)}")

    return num_cameras, camera_models


class ImageReconstructionDataset(Dataset):
    """
    PyTorch Dataset for compressed image reconstruction with camera model conditioning.
    """

    def __init__(self, compressed_base_path=None, uncompressed_base_path=None, transform=None, target_size=(256, 256)):
        """
        Initialize the dataset.

        Parameters:
        -----------
        compressed_base_path : str, optional
            Path to the compressed images directory. If None, uses environment variable.
        uncompressed_base_path : str, optional
            Path to the uncompressed images directory. If None, uses environment variable.
        transform : torchvision.transforms, optional
            Optional transforms to be applied on the images
        target_size : tuple
            Size to resize images to (height, width)
        """
        # Use environment variables if paths are not provided
        self.compressed_base_path = compressed_base_path or os.path.join(EXTERNAL_DRIVE, COMPRESSED_IMAGES_DIR)
        self.uncompressed_base_path = uncompressed_base_path or os.path.join(EXTERNAL_DRIVE, ORIGINAL_IMAGES_DIR)
        self.transform = transform
        self.target_size = target_size

        # Get the list of camera models and create a mapping to indices
        self.num_cameras, self.camera_models = count_cameras(self.compressed_base_path)
        self.camera_to_idx = {model: idx for idx, model in enumerate(self.camera_models)}

        # Default transform if none is provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # Create a list of all image pairs
        self.image_pairs = []
        # Use quality levels from environment variables
        quality_levels = QUALITY_LEVELS

        for camera in self.camera_models:
            for quality in quality_levels:
                compressed_folder = os.path.join(self.compressed_base_path, camera, str(quality))
                uncompressed_folder = os.path.join(self.uncompressed_base_path, camera)

                if not os.path.exists(compressed_folder) or not os.path.exists(uncompressed_folder):
                    continue

                # Get image filenames in this folder
                for img_name in os.listdir(compressed_folder):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        compressed_img_path = os.path.join(compressed_folder, img_name)

                        # Find corresponding uncompressed image (might have different extension)
                        uncompressed_img_name = os.path.splitext(img_name)[0]
                        found = False
                        for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.DNG']:
                            uncompressed_path = os.path.join(uncompressed_folder, uncompressed_img_name + ext)
                            if os.path.exists(uncompressed_path):
                                found = True
                                break

                        if found:
                            self.image_pairs.append({
                                'compressed_path': compressed_img_path,
                                'uncompressed_path': uncompressed_path,
                                'camera_model': camera,
                                'camera_idx': self.camera_to_idx[camera],
                                'quality_level': quality
                            })

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
        --------
        tuple: (dict, tensor)
            - dict contains input image tensor and condition tensors (camera_idx, quality_level)
            - tensor is the target uncompressed image
        """
        pair = self.image_pairs[idx]

        # Load compressed image
        compressed_img = Image.open(pair['compressed_path']).convert('RGB')
        compressed_tensor = self.transform(compressed_img)

        # Load uncompressed (target) image
        uncompressed_img = Image.open(pair['uncompressed_path']).convert('RGB')
        target_tensor = self.transform(uncompressed_img)

        # Create the input dictionary with image and condition information
        input_dict = {
            'compressed_image': compressed_tensor,
            'camera_idx': torch.tensor(pair['camera_idx'], dtype=torch.long),
            'quality_level': torch.tensor(pair['quality_level'], dtype=torch.long)
        }

        return input_dict, target_tensor
