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
SPN_IMAGES_DIR = os.getenv('SPN_IMAGES_DIR')
QUALITY_LEVELS = [int(q) for q in os.getenv('QUALITY_LEVELS').split(',')]

def create_spn_pairs(compressed_path, spn_path):
    """
    Create pairs of compressed images with their SPNs and target camera SPN.
    
    Args:
        compressed_path (str): Path to compressed images directory
        spn_path (str): Path to SPN images directory
        
    Returns:
        list: List of dictionaries containing image pairs and their SPNs
    """
    pairs = []

    # Iterate through each camera model
    for camera_model in os.listdir(compressed_path):
        camera_compressed_path = os.path.join(compressed_path, camera_model)
        camera_spn_path = os.path.join(spn_path, camera_model)

        # Skip if not a directory
        if not os.path.isdir(camera_compressed_path) or not os.path.isdir(camera_spn_path):
            continue

        # Get the camera's averaged SPN (target)
        camera_spn_path = os.path.join(camera_spn_path, 'original', 'camera_SPN.png')
        if not os.path.exists(camera_spn_path):
            print(f"Warning: No camera SPN found for {camera_model}")
            continue

        # For each quality level
        for quality in QUALITY_LEVELS:
            quality_dir = os.path.join(camera_compressed_path, str(quality))
            spn_quality_dir = os.path.join(camera_spn_path, 'compressed', str(quality))

            if not os.path.exists(quality_dir) or not os.path.exists(spn_quality_dir):
                continue

            # Get all compressed images in this quality level
            for img_name in os.listdir(quality_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg')):
                    # Paths for the compressed image and its SPN
                    compressed_img_path = os.path.join(quality_dir, img_name)
                    compressed_spn_path = os.path.join(spn_quality_dir, f"{os.path.splitext(img_name)[0]}_SPN.png")

                    if os.path.exists(compressed_spn_path):
                        pairs.append({
                            'compressed_image': compressed_img_path,
                            'compressed_spn': compressed_spn_path,
                            'camera_spn': camera_spn_path,  # This is the averaged camera SPN
                            'quality_level': quality
                        })

    return pairs

class SPNRestorationDataset(Dataset):
    """
    PyTorch Dataset for SPN restoration from compressed images.
    The target is the camera's averaged SPN.
    """

    def __init__(self, compressed_base_path=None, spn_base_path=None, transform=None, target_size=(256, 256)):
        """
        Initialize the dataset.

        Parameters:
        -----------
        compressed_base_path : str, optional
            Path to the compressed images directory. If None, uses environment variable.
        spn_base_path : str, optional
            Path to the SPN images directory. If None, uses environment variable.
        transform : torchvision.transforms, optional
            Optional transforms to be applied on the images
        target_size : tuple
            Size to resize images to (height, width)
        """
        # Use environment variables if paths are not provided
        self.compressed_base_path = compressed_base_path or os.path.join(EXTERNAL_DRIVE, COMPRESSED_IMAGES_DIR)
        self.spn_base_path = spn_base_path or os.path.join(EXTERNAL_DRIVE, SPN_IMAGES_DIR)
        self.transform = transform
        self.target_size = target_size

        # Default transform if none is provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # Create the list of image pairs with their SPNs
        self.image_pairs = create_spn_pairs(self.compressed_base_path, self.spn_base_path)

        print(f"Created dataset with {len(self.image_pairs)} samples")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
        --------
        tuple: (dict, tensor)
            - dict contains input image tensor, compressed SPN tensor, and quality level
            - tensor is the target camera SPN (averaged)
        """
        pair = self.image_pairs[idx]

        # Load compressed image
        compressed_img = Image.open(pair['compressed_image']).convert('RGB')
        compressed_tensor = self.transform(compressed_img)

        # Load compressed SPN
        compressed_spn = Image.open(pair['compressed_spn']).convert('L')  # Convert to grayscale
        compressed_spn_tensor = transforms.ToTensor()(compressed_spn)

        # Load camera SPN (target) - this is the averaged SPN
        camera_spn = Image.open(pair['camera_spn']).convert('L')  # Convert to grayscale
        camera_spn_tensor = transforms.ToTensor()(camera_spn)

        # Create the input dictionary
        input_dict = {
            'compressed_image': compressed_tensor,
            'compressed_spn': compressed_spn_tensor,
            'quality_level': torch.tensor(pair['quality_level'], dtype=torch.long)
        }

        return input_dict, camera_spn_tensor
