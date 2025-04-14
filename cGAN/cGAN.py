import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import os
import multiprocessing
import time
from dotenv import load_dotenv

from Dataset.generate_dataset import ImageReconstructionDataset
from cGAN.generator import Generator
from cGAN.discriminator import Discriminator

# Load environment variables
load_dotenv()

# Get paths from environment variables
EXTERNAL_DRIVE = os.getenv('EXTERNAL_DRIVE')
COMPRESSED_IMAGES_DIR = os.getenv('COMPRESSED_IMAGES_DIR')
ORIGINAL_IMAGES_DIR = os.getenv('ORIGINAL_IMAGES_DIR')

# Binary Cross-Entropy Loss and Cross-Entropy Loss for classification
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()

def train_discriminator(discriminator, generator, real_images, compressed_images, camera_labels, discriminator_optimizer):
    """
    Train the discriminator with real and fake data.

    Returns:
    --------
    dict: Dictionary containing discriminator losses.
    """
    discriminator.zero_grad()

    # Real images
    real_labels = torch.ones(real_images.size(0)).cuda()
    fake_labels = torch.zeros(compressed_images.size(0)).cuda()

    real_authenticity, real_camera_logits = discriminator(real_images)
    real_authenticity_loss = bce_loss(real_authenticity, real_labels)
    real_camera_loss = ce_loss(real_camera_logits, camera_labels)
    real_loss = real_authenticity_loss + real_camera_loss
    real_loss.backward()

    # Fake images (from generator)
    fake_images = generator(compressed_images)
    fake_authenticity, _ = discriminator(fake_images.detach())
    fake_authenticity_loss = bce_loss(fake_authenticity, fake_labels)
    fake_loss = fake_authenticity_loss
    fake_loss.backward()

    # Optimizer step
    discriminator_optimizer.step()

    return {
        "real_authenticity_loss": real_authenticity_loss.item(),
        "real_camera_loss": real_camera_loss.item(),
        "fake_authenticity_loss": fake_authenticity_loss.item()
    }

def train_generator(generator, discriminator, compressed_images, camera_labels, generator_optimizer):
    """
    Train the generator to fool the discriminator.

    Returns:
    --------
    dict: Dictionary containing generator loss.
    """
    generator.zero_grad()

    # Generate fake images
    fake_images = generator(compressed_images)
    fake_authenticity, fake_camera_logits = discriminator(fake_images)

    # Authenticity loss (want discriminator to predict 'real' for fake images)
    real_labels = torch.ones(fake_authenticity.size(0)).cuda()
    authenticity_loss = bce_loss(fake_authenticity, real_labels)

    # Camera classification loss
    camera_loss = ce_loss(fake_camera_logits, camera_labels)

    # Total generator loss
    generator_loss = authenticity_loss + camera_loss
    generator_loss.backward()

    # Optimizer step
    generator_optimizer.step()

    return {
        "authenticity_loss": authenticity_loss.item(),
        "camera_loss": camera_loss.item()
    }

def train_cgan(config):
    """
    Main training function for the cGAN model.

    Args:
        config (dict): Configuration dictionary containing:
            - batch_size: Batch size for training
            - num_epochs: Number of training epochs
            - learning_rate: Learning rate for optimizers
            - num_workers: Number of workers for data loading
    """
    print("========== STARTING cGAN TRAINING ==========")
    multiprocessing.freeze_support()
    print("Multiprocessing freeze_support initialized")

    # Create the dataset
    compressed_path = os.path.join(EXTERNAL_DRIVE, COMPRESSED_IMAGES_DIR)
    uncompressed_path = os.path.join(EXTERNAL_DRIVE, ORIGINAL_IMAGES_DIR)
    print(f"Setting up paths:\n  - Compressed: {compressed_path}\n  - Uncompressed: {uncompressed_path}")

    # Check if the directories exist
    if not os.path.exists(compressed_path):
        raise FileNotFoundError(f"Compressed images directory not found: {compressed_path}")
    if not os.path.exists(uncompressed_path):
        raise FileNotFoundError(f"Original images directory not found: {uncompressed_path}")
    print("✓ Directory paths verified")

    print("Creating dataset object...")
    dataset = ImageReconstructionDataset(
        compressed_base_path=compressed_path,
        uncompressed_base_path=uncompressed_path,
        target_size=(256, 256),
        transform=None
    )
    print(f"✓ Dataset created with {len(dataset)} samples")

    # Create data loader
    print(f"Initializing DataLoader with batch size {config['batch_size']}...")
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    print("✓ DataLoader initialized")

    print("Initializing models and moving to CUDA...")
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    print("✓ Models created and moved to CUDA")

    print("Setting up optimizers...")
    generator_optimizer = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
    print("✓ Optimizers configured")

    print("\n" + "=" * 50)
    print("BEGINNING TRAINING")
    print("=" * 50)

    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\n{'=' * 20} EPOCH {epoch + 1}/{config['num_epochs']} {'=' * 20}")
        epoch_start_time = time.time()

        for i, data in enumerate(data_loader):
            batch_start_time = time.time()
            print(f"\n--- Batch {i + 1}/{len(data_loader)} ---")

            # Unpack the tuple returned by the dataset
            print("  • Unpacking data batch...")
            input_dict, target_tensor = data

            # Get data from input_dict and move to cuda
            compressed_images = input_dict['compressed_image'].cuda()
            camera_labels = input_dict['camera_idx'].cuda()
            real_images = target_tensor.cuda()

            # Train discriminator
            print("  • Training discriminator...")
            disc_losses = train_discriminator(discriminator, generator, real_images, compressed_images, camera_labels, discriminator_optimizer)
            disc_loss_sum = sum(disc_losses.values())
            print(f"  ✓ Discriminator training complete - Loss: {disc_loss_sum:.4f}")

            # Train generator
            print("  • Training generator...")
            gen_losses = train_generator(generator, discriminator, compressed_images, camera_labels, generator_optimizer)
            gen_loss_sum = sum(gen_losses.values())
            print(f"  ✓ Generator training complete - Loss: {gen_loss_sum:.4f}")

            batch_time = time.time() - batch_start_time

            # Print progress
            if i % 10 == 0:
                print(f"\nBatch {i + 1}/{len(data_loader)} Summary:")
                print(f"  - D_loss={disc_loss_sum:.4f}")
                print(f"    - Details: {', '.join([f'{k}={v:.4f}' for k, v in disc_losses.items()])}")
                print(f"  - G_loss={gen_loss_sum:.4f}")
                print(f"    - Details: {', '.join([f'{k}={v:.4f}' for k, v in gen_losses.items()])}")
                print(f"  - Batch processing time: {batch_time:.2f} seconds")

        epoch_time = time.time() - epoch_start_time
        print(f"\n{'=' * 20} EPOCH {epoch + 1} COMPLETE {'=' * 20}")
        print(f"Time taken: {epoch_time:.2f} seconds")

if __name__ == "__main__":
    # Default configuration
    config = {
        'batch_size': 16,
        'num_epochs': 20,
        'learning_rate': 0.0002,
        'num_workers': 4
    }
    
    train_cgan(config)
