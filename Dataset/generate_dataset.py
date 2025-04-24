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
    Limits the number of pairs to 500 per camera model.
    
    Args:
        compressed_path (str): Path to compressed images directory
        spn_path (str): Path to SPN images directory
        
    Returns:
        list: List of dictionaries containing image pairs and their SPNs
    """
    pairs = []
    print(f"Scanning for image pairs...")
    pairs_per_camera = {}  # Track pairs for each camera

    # Iterate through each camera model
    for camera_model in os.listdir(compressed_path):
        camera_compressed_path = os.path.join(compressed_path, camera_model)
        camera_spn_base_path = os.path.join(spn_path, camera_model)
        pairs_per_camera[camera_model] = []

        # Skip if not a directory
        if not os.path.isdir(camera_compressed_path) or not os.path.isdir(camera_spn_base_path):
            continue

        # Get the camera's averaged SPN (target)
        camera_spn_path = os.path.join(camera_spn_base_path, 'original', 'camera_SPN.png')
        
        if not os.path.exists(camera_spn_path):
            continue

        # For each quality level
        for quality in QUALITY_LEVELS:
            quality_dir = os.path.join(camera_compressed_path, str(quality))
            spn_quality_dir = os.path.join(camera_spn_base_path, 'compressed', str(quality))

            if not os.path.exists(quality_dir) or not os.path.exists(spn_quality_dir):
                continue

            # Get all compressed images in this quality level
            compressed_images = [f for f in os.listdir(quality_dir) if f.lower().endswith(('.jpg', '.jpeg'))]

            for img_name in compressed_images:
                # Stop if we already have 500 pairs for this camera
                if len(pairs_per_camera[camera_model]) >= 500:
                    break
                    
                # Paths for the compressed image and its SPN
                compressed_img_path = os.path.join(quality_dir, img_name)
                compressed_spn_path = os.path.join(spn_quality_dir, f"{os.path.splitext(img_name)[0]}_SPN.png")

                if os.path.exists(compressed_spn_path):
                    pair = {
                        'compressed_image': compressed_img_path,
                        'compressed_spn': compressed_spn_path,
                        'camera_spn': camera_spn_path,  # This is the averaged camera SPN
                        'quality_level': quality
                    }
                    pairs_per_camera[camera_model].append(pair)

            # Break quality level loop if we have enough pairs
            if len(pairs_per_camera[camera_model]) >= 500:
                break

    # Combine all pairs
    for camera_model, camera_pairs in pairs_per_camera.items():
        print(f"Found {len(camera_pairs)} pairs for camera {camera_model}")
        pairs.extend(camera_pairs)

    print(f"Total pairs found: {len(pairs)}")
    return pairs

class SPNRestorationDataset(Dataset):
    """
    PyTorch Dataset for SPN restoration from compressed images.
    The target is the camera's averaged SPN.
    Uses random patch extraction to handle images of different sizes.
    """

    def __init__(self, compressed_base_path=None, spn_base_path=None, transform=None, patch_size=(256, 256)):
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
        patch_size : tuple
            Size of patches to extract (height, width)
        """
        # Use environment variables if paths are not provided
        self.compressed_base_path = compressed_base_path or os.path.join(EXTERNAL_DRIVE, COMPRESSED_IMAGES_DIR)
        self.spn_base_path = spn_base_path or os.path.join(EXTERNAL_DRIVE, SPN_IMAGES_DIR)
        self.transform = transform
        self.patch_size = patch_size

        # Default transform if none is provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # Create the list of image pairs with their SPNs
        self.image_pairs = create_spn_pairs(self.compressed_base_path, self.spn_base_path)

        print(f"Created dataset with {len(self.image_pairs)} samples")

    def extract_patch(self, img, patch_size, i, j):
        """
        Extract a patch from an image at position (i, j).
        
        Parameters:
        -----------
        img : PIL.Image
            Input image
        patch_size : tuple
            Size of patch to extract (height, width)
        i : int
            Top coordinate
        j : int
            Left coordinate
            
        Returns:
        --------
        PIL.Image
            Extracted patch
        """
        return img.crop((j, i, j + patch_size[1], i + patch_size[0]))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Extracts random patches of the same size from all images.

        Returns:
        --------
        tuple: (dict, tensor)
            - dict contains input image tensor, compressed SPN tensor, and quality level
            - tensor is the target camera SPN (averaged)
        """
        pair = self.image_pairs[idx]

        # Load compressed image
        compressed_img = Image.open(pair['compressed_image']).convert('RGB')
        w, h = compressed_img.size

        # Load compressed SPN and camera SPN
        compressed_spn = Image.open(pair['compressed_spn']).convert('L')
        camera_spn = Image.open(pair['camera_spn']).convert('L')

        # Calculate valid patch coordinates
        max_i = h - self.patch_size[0]
        max_j = w - self.patch_size[1]

        if max_i < 0 or max_j < 0:
            # If image is smaller than patch size, resize the image
            new_h = max(h, self.patch_size[0])
            new_w = max(w, self.patch_size[1])
            compressed_img = compressed_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            compressed_spn = compressed_spn.resize((new_w, new_h), Image.Resampling.LANCZOS)
            camera_spn = camera_spn.resize((new_w, new_h), Image.Resampling.LANCZOS)
            max_i = new_h - self.patch_size[0]
            max_j = new_w - self.patch_size[1]

        # Generate random patch coordinates
        i = torch.randint(0, max_i + 1, (1,)).item()
        j = torch.randint(0, max_j + 1, (1,)).item()

        # Extract patches
        compressed_patch = self.extract_patch(compressed_img, self.patch_size, i, j)
        compressed_spn_patch = self.extract_patch(compressed_spn, self.patch_size, i, j)
        camera_spn_patch = self.extract_patch(camera_spn, self.patch_size, i, j)

        # Apply transforms
        compressed_tensor = self.transform(compressed_patch)
        compressed_spn_tensor = transforms.ToTensor()(compressed_spn_patch)
        camera_spn_tensor = transforms.ToTensor()(camera_spn_patch)

        # Create the input dictionary
        input_dict = {
            'compressed_image': compressed_tensor,
            'compressed_spn': compressed_spn_tensor,
            'quality_level': torch.tensor(pair['quality_level'], dtype=torch.long),
            'camera_idx': torch.tensor(0, dtype=torch.long)  # Added camera index
        }

        return input_dict, camera_spn_tensor
