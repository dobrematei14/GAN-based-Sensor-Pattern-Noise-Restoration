import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import os
import multiprocessing
import time
from dotenv import load_dotenv
from datetime import datetime
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Dataset.generate_dataset import SPNDataset
from cGAN.generator import Generator
from cGAN.discriminator import Discriminator

# Load environment variables
load_dotenv()

# Get paths from environment variables
DRIVE = os.getenv('EXTERNAL_DRIVE')
COMP_DIR = os.getenv('COMPRESSED_IMAGES_DIR')
ORIG_DIR = os.getenv('ORIGINAL_IMAGES_DIR')
SPN_DIR = os.getenv('SPN_IMAGES_DIR')

# Validate environment variables
if not all([DRIVE, COMP_DIR, ORIG_DIR, SPN_DIR]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# Create models directory structure in project root
MODELS_DIR = os.path.join(project_root, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Loss functions
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()
l1_loss = nn.L1Loss()

def frequency_domain_loss(output, target):
    """Compute frequency domain consistency loss"""
    output_fft = torch.fft.fft2(output)
    target_fft = torch.fft.fft2(target)
    
    # Magnitude loss
    mag_loss = l1_loss(torch.abs(output_fft), torch.abs(target_fft))
    
    # Phase loss
    phase_loss = l1_loss(torch.angle(output_fft), torch.angle(target_fft))
    
    return mag_loss + phase_loss

def train_discriminator(discriminator, generator, real_images, compressed_images, compressed_spn, compression_level, camera_labels, discriminator_optimizer):
    """Train the discriminator with real and fake data."""
    discriminator.zero_grad()
    
    batch_size = real_images.size(0)
    real_labels = torch.ones(batch_size, 1).cuda()
    fake_labels = torch.zeros(batch_size, 1).cuda()
    
    # Convert grayscale to RGB by repeating the channel
    real_images_rgb = real_images.repeat(1, 3, 1, 1)
    
    # Real images
    real_authenticity, real_camera_logits, real_spn = discriminator(real_images_rgb)
    real_authenticity = real_authenticity.view(batch_size, 1)
    real_authenticity_loss = bce_loss(real_authenticity, real_labels)
    real_camera_loss = ce_loss(real_camera_logits, camera_labels)
    real_spn_loss = bce_loss(real_spn, torch.ones(batch_size, 1).cuda())
    real_loss = real_authenticity_loss + real_camera_loss + real_spn_loss
    real_loss.backward()
    
    # Fake images
    fake_images = generator(compressed_images, compressed_spn, compression_level)
    # Convert fake images to RGB if they're grayscale
    if fake_images.size(1) == 1:
        fake_images = fake_images.repeat(1, 3, 1, 1)
    fake_authenticity, fake_camera_logits, fake_spn = discriminator(fake_images.detach())
    fake_authenticity = fake_authenticity.view(batch_size, 1)
    fake_authenticity_loss = bce_loss(fake_authenticity, fake_labels)
    fake_camera_loss = ce_loss(fake_camera_logits, camera_labels)
    fake_spn_loss = bce_loss(fake_spn, torch.zeros(batch_size, 1).cuda())
    fake_loss = fake_authenticity_loss + fake_camera_loss + fake_spn_loss
    fake_loss.backward()
    
    discriminator_optimizer.step()
    
    return {
        "real_authenticity_loss": real_authenticity_loss.item(),
        "real_camera_loss": real_camera_loss.item(),
        "real_spn_loss": real_spn_loss.item(),
        "fake_authenticity_loss": fake_authenticity_loss.item(),
        "fake_camera_loss": fake_camera_loss.item(),
        "fake_spn_loss": fake_spn_loss.item()
    }

def train_generator(generator, discriminator, compressed_images, compressed_spn, compression_level, camera_labels, real_images, generator_optimizer):
    """Train the generator to fool the discriminator and restore SPN."""
    generator.zero_grad()
    
    batch_size = compressed_images.size(0)
    
    # Generate fake images
    fake_images = generator(compressed_images, compressed_spn, compression_level)
    # Convert fake images to RGB if they're grayscale
    if fake_images.size(1) == 1:
        fake_images = fake_images.repeat(1, 3, 1, 1)
    fake_authenticity, fake_camera_logits, fake_spn = discriminator(fake_images)
    
    # Adversarial losses
    real_labels = torch.ones(batch_size, 1).cuda()
    fake_authenticity = fake_authenticity.view(batch_size, 1)
    authenticity_loss = bce_loss(fake_authenticity, real_labels)
    camera_loss = ce_loss(fake_camera_logits, camera_labels)
    spn_loss = bce_loss(fake_spn, torch.ones(batch_size, 1).cuda())
    
    # Convert real images to RGB for reconstruction loss
    real_images_rgb = real_images.repeat(1, 3, 1, 1)
    
    # Reconstruction losses
    l1_reconstruction_loss = l1_loss(fake_images, real_images_rgb)
    freq_loss = frequency_domain_loss(fake_images, real_images_rgb)
    
    # Total generator loss
    generator_loss = authenticity_loss + camera_loss + spn_loss + 10.0 * l1_reconstruction_loss + 5.0 * freq_loss
    generator_loss.backward()
    
    generator_optimizer.step()
    
    return {
        "authenticity_loss": authenticity_loss.item(),
        "camera_loss": camera_loss.item(),
        "spn_loss": spn_loss.item(),
        "l1_reconstruction_loss": l1_reconstruction_loss.item(),
        "freq_loss": freq_loss.item()
    }

def create_model_directory():
    """Create a new directory for this training run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(MODELS_DIR, f"training_run_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def save_checkpoint(generator, discriminator, generator_optimizer, discriminator_optimizer, epoch, config, model_dir):
    """Save model checkpoint"""
    checkpoint = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'generator_optimizer_state_dict': generator_optimizer.state_dict(),
        'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
        'epoch': epoch,
        'config': config
    }
    
    checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")

def load_checkpoint(generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['config']

def train_cgan(config):
    """Main training function for the cGAN model."""
    print("========== STARTING cGAN TRAINING ==========")
    multiprocessing.freeze_support()
    print("Multiprocessing freeze_support initialized")
    
    # Create model directory for this training run
    model_dir = create_model_directory()
    print(f"Created model directory: {model_dir}")
    
    # Create the dataset
    compressed_path = os.path.join(DRIVE, COMP_DIR)
    spn_path = os.path.join(DRIVE, SPN_DIR)
    print(f"Setting up paths:\n  - Compressed: {compressed_path}\n  - SPN: {spn_path}")
    
    # Check if the directories exist
    if not os.path.exists(compressed_path):
        raise FileNotFoundError(f"Compressed images directory not found: {compressed_path}")
    if not os.path.exists(spn_path):
        raise FileNotFoundError(f"SPN images directory not found: {spn_path}")
    print("✓ Directory paths verified")
    
    # Check directory structure
    print("\nChecking directory structure...")
    for camera_model in os.listdir(compressed_path):
        camera_compressed_path = os.path.join(compressed_path, camera_model)
        camera_spn_path = os.path.join(spn_path, camera_model)
        
        if os.path.isdir(camera_compressed_path) and os.path.isdir(camera_spn_path):
            print(f"✓ Found camera model: {camera_model}")
            print(f"  - Compressed images: {camera_compressed_path}")
            print(f"  - SPN images: {camera_spn_path}")
        else:
            print(f"⚠ Warning: Incomplete directory structure for {camera_model}")
    
    print("\nCreating dataset object...")
    dataset = SPNDataset(
        patch_size=(256, 256)  # Or any other size that fits in GPU memory
    )
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty! Please check your directory structure and image files.")
    
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
    
    # Calculate checkpoint intervals
    total_epochs = config['num_epochs']
    checkpoint_intervals = [int(total_epochs * p) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    
    print("\n" + "=" * 50)
    print("BEGINNING TRAINING")
    print("=" * 50)
    
    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Initialize epoch metrics
        epoch_d_losses = []
        epoch_g_losses = []
        
        # Process each batch
        for batch_idx, (input_dict, target_spn) in enumerate(data_loader):
            # Move data to GPU
            compressed_images = input_dict['comp_img'].cuda()
            compressed_spn = input_dict['comp_spn'].cuda()
            compression_level = input_dict['quality'].cuda()
            camera_labels = input_dict['cam_idx'].cuda()
            real_images = target_spn.cuda()
            
            # Train discriminator
            d_losses = train_discriminator(
                discriminator, generator, real_images, compressed_images,
                compressed_spn, compression_level, camera_labels, discriminator_optimizer
            )
            
            # Train generator
            g_losses = train_generator(
                generator, discriminator, compressed_images, compressed_spn,
                compression_level, camera_labels, real_images, generator_optimizer
            )
            
            # Store losses
            epoch_d_losses.append(d_losses)
            epoch_g_losses.append(g_losses)
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(data_loader)}")
                print(f"D Loss: {sum(d_losses.values()):.4f}")
                print(f"G Loss: {sum(g_losses.values()):.4f}")
        
        # Calculate and print epoch metrics
        avg_d_loss = sum(sum(loss.values()) for loss in epoch_d_losses) / len(epoch_d_losses)
        avg_g_loss = sum(sum(loss.values()) for loss in epoch_g_losses) / len(epoch_g_losses)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Average D Loss: {avg_d_loss:.4f}")
        print(f"Average G Loss: {avg_g_loss:.4f}")
        
        # Save checkpoint if needed
        if (epoch + 1) in checkpoint_intervals:
            save_checkpoint(
                generator, discriminator, generator_optimizer,
                discriminator_optimizer, epoch + 1, config, model_dir
            )
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    
    return {
        'model_dir': model_dir,
        'final_d_loss': avg_d_loss,
        'final_g_loss': avg_g_loss
    }

if __name__ == "__main__":
    # Configuration optimized for 1000 image pairs (500 per camera)
    config = {
        'batch_size': 32,        # Increased batch size since we have less data
        'num_epochs': 100,       # Fewer epochs needed for smaller dataset
        'learning_rate': 0.0002, # Slightly higher learning rate for faster convergence
        'num_workers': 4         # Keep same number of workers
    }
    
    train_cgan(config)
