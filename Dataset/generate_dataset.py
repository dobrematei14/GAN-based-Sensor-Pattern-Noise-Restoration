import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get paths from environment variables
DRIVE = os.getenv('EXTERNAL_DRIVE')
COMP_DIR = os.getenv('COMPRESSED_IMAGES_DIR')
ORIG_DIR = os.getenv('ORIGINAL_IMAGES_DIR')
SPN_DIR = os.getenv('SPN_IMAGES_DIR')
QUALITIES = [int(q) for q in os.getenv('QUALITY_LEVELS').split(',')]

def create_pairs(comp_path, spn_path):
    """Creates pairs of compressed images with their corresponding SPNs and target camera SPN.
    Limits to 500 pairs per camera model."""
    pairs = []
    cam_pairs = {}

    for cam in os.listdir(comp_path):
        cam_comp_path = os.path.join(comp_path, cam)
        cam_spn_path = os.path.join(spn_path, cam)
        cam_pairs[cam] = []

        if not os.path.isdir(cam_comp_path) or not os.path.isdir(cam_spn_path):
            continue

        cam_spn = os.path.join(cam_spn_path, 'original', 'camera_SPN.png')
        
        if not os.path.exists(cam_spn):
            continue

        for q in QUALITIES:
            q_dir = os.path.join(cam_comp_path, str(q))
            spn_q_dir = os.path.join(cam_spn_path, 'compressed', str(q))

            if not os.path.exists(q_dir) or not os.path.exists(spn_q_dir):
                continue

            imgs = [f for f in os.listdir(q_dir) if f.lower().endswith(('.jpg', '.jpeg'))]

            for img in imgs:
                if len(cam_pairs[cam]) >= 500:
                    break
                    
                comp_img = os.path.join(q_dir, img)
                comp_spn = os.path.join(spn_q_dir, f"{os.path.splitext(img)[0]}_SPN.png")

                if os.path.exists(comp_spn):
                    pair = {
                        'comp_img': comp_img,
                        'comp_spn': comp_spn,
                        'cam_spn': cam_spn,
                        'quality': q
                    }
                    cam_pairs[cam].append(pair)

            if len(cam_pairs[cam]) >= 500:
                break

    for cam, cam_pair in cam_pairs.items():
        pairs.extend(cam_pair)

    return pairs

class SPNDataset(Dataset):
    """PyTorch Dataset for SPN restoration from compressed images.
    Handles random patch extraction and image transformations."""
    
    def __init__(self, comp_path=None, spn_path=None, transform=None, patch_size=(256, 256)):
        """Initializes the dataset with paths and transformation settings."""
        self.comp_path = comp_path or os.path.join(DRIVE, COMP_DIR)
        self.spn_path = spn_path or os.path.join(DRIVE, SPN_DIR)
        self.transform = transform
        self.patch_size = patch_size

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.pairs = create_pairs(self.comp_path, self.spn_path)

    def get_patch(self, img, size, i, j):
        """Extracts a patch from an image at specified coordinates."""
        return img.crop((j, i, j + size[1], i + size[0]))

    def __len__(self):
        """Returns the total number of image pairs in the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Returns a random patch from the image pair at the given index.
        Includes compressed image, SPN, and quality level information."""
        pair = self.pairs[idx]

        comp_img = Image.open(pair['comp_img']).convert('RGB')
        w, h = comp_img.size

        comp_spn = Image.open(pair['comp_spn']).convert('L')
        cam_spn = Image.open(pair['cam_spn']).convert('L')

        max_i = h - self.patch_size[0]
        max_j = w - self.patch_size[1]

        if max_i < 0 or max_j < 0:
            new_h = max(h, self.patch_size[0])
            new_w = max(w, self.patch_size[1])
            comp_img = comp_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            comp_spn = comp_spn.resize((new_w, new_h), Image.Resampling.LANCZOS)
            cam_spn = cam_spn.resize((new_w, new_h), Image.Resampling.LANCZOS)
            max_i = new_h - self.patch_size[0]
            max_j = new_w - self.patch_size[1]

        i = torch.randint(0, max_i + 1, (1,)).item()
        j = torch.randint(0, max_j + 1, (1,)).item()

        comp_patch = self.get_patch(comp_img, self.patch_size, i, j)
        comp_spn_patch = self.get_patch(comp_spn, self.patch_size, i, j)
        cam_spn_patch = self.get_patch(cam_spn, self.patch_size, i, j)

        comp_tensor = self.transform(comp_patch)
        comp_spn_tensor = transforms.ToTensor()(comp_spn_patch)
        cam_spn_tensor = transforms.ToTensor()(cam_spn_patch)

        input_dict = {
            'comp_img': comp_tensor,
            'comp_spn': comp_spn_tensor,
            'quality': torch.tensor(pair['quality'], dtype=torch.long),
            'cam_idx': torch.tensor(0, dtype=torch.long)
        }

        return input_dict, cam_spn_tensor
