import torch
from torch.utils.data import DataLoader
from Dataset.generate_dataset import ImageReconstructionDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import multiprocessing
import time

# Binary Cross-Entropy Loss and Cross-Entropy Loss for classification
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()


class Discriminator(nn.Module):
    def __init__(self, input_channels=3, num_cameras=10):  # 3 (image)
        super(Discriminator, self).__init__()
        self.num_cameras = num_cameras

        # Feature extraction
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)

        # Real/Fake classification head
        self.authenticity_head = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),  # Global pooling to get single prediction
            nn.Sigmoid()  # Add sigmoid to ensure outputs are between 0 and 1
        )

        # Camera identification head
        self.camera_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=4, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),  # Global pooling
            nn.Flatten(),
            nn.Linear(256, num_cameras)  # Outputs logits for each camera
        )

        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

    def forward(self, image):
        # Shared feature extraction
        x = F.leaky_relu(self.conv1(image), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        features = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)

        # Get authenticity prediction (real/fake)
        authenticity = self.authenticity_head(features)

        # Get camera prediction
        camera_logits = self.camera_head(features)

        return authenticity.squeeze(), camera_logits


class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):  # SPN restoration outputs full restored image
        super(Generator, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.enc5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)

        # Decoder with skip connections
        self.dec1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.dec4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.dec5 = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)

        self.bn_enc2 = nn.BatchNorm2d(128)
        self.bn_enc3 = nn.BatchNorm2d(256)
        self.bn_enc4 = nn.BatchNorm2d(512)
        self.bn_enc5 = nn.BatchNorm2d(512)

        self.bn_dec1 = nn.BatchNorm2d(512)
        self.bn_dec2 = nn.BatchNorm2d(256)
        self.bn_dec3 = nn.BatchNorm2d(128)
        self.bn_dec4 = nn.BatchNorm2d(64)

    def forward(self, compressed_img):
        # Encoder
        e1 = F.leaky_relu(self.enc1(compressed_img), 0.2)
        e2 = F.leaky_relu(self.bn_enc2(self.enc2(e1)), 0.2)
        e3 = F.leaky_relu(self.bn_enc3(self.enc3(e2)), 0.2)
        e4 = F.leaky_relu(self.bn_enc4(self.enc4(e3)), 0.2)
        e5 = F.leaky_relu(self.bn_enc5(self.enc5(e4)), 0.2)

        # Decoder with skip connections
        d1 = F.relu(self.bn_dec1(self.dec1(e5)))
        d1 = torch.cat([d1, e4], 1)
        d2 = F.relu(self.bn_dec2(self.dec2(d1)))
        d2 = torch.cat([d2, e3], 1)
        d3 = F.relu(self.bn_dec3(self.dec3(d2)))
        d3 = torch.cat([d3, e2], 1)
        d4 = F.relu(self.bn_dec4(self.dec4(d3)))
        d4 = torch.cat([d4, e1], 1)

        # Output full restored image
        restored_image = torch.tanh(self.dec5(d4))  # Add tanh for output normalization

        return restored_image


def train_discriminator(discriminator, generator, real_images, compressed_images, camera_labels):
    """
    Train the discriminator with real and fake data.

    Returns:
    --------
    dict: Dictionary containing discriminator losses.
    """
    discriminator.zero_grad()

    # Real images
    real_labels = torch.ones(real_images.size(0)).cuda()  # Changed to match output shape
    fake_labels = torch.zeros(compressed_images.size(0)).cuda()  # Changed to match output shape

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


def train_generator(generator, discriminator, compressed_images, camera_labels):
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
    real_labels = torch.ones(fake_authenticity.size(0)).cuda()  # Changed to match output shape
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


if __name__ == "__main__":
    print("========== STARTING cGAN TRAINING ==========")
    # Add freeze_support for Windows
    multiprocessing.freeze_support()
    print("Multiprocessing freeze_support initialized")

    # Create the dataset
    compressed_path = "F:\\Images\\Compressed"
    uncompressed_path = "F:\\Images\\Original"
    print(f"Setting up paths:\n  - Compressed: {compressed_path}\n  - Uncompressed: {uncompressed_path}")

    # Check if the directories exist
    if not os.path.exists(compressed_path):
        raise FileNotFoundError(f"Compressed images directory not found: {compressed_path}")
    if not os.path.exists(uncompressed_path):
        raise FileNotFoundError(f"Original images directory not found: {uncompressed_path}")
    print("✓ Directory paths verified")

    print(f"Compressed directory contents: {os.listdir(compressed_path)[:5]}...")
    print(f"Uncompressed directory contents: {os.listdir(uncompressed_path)[:5]}...")

    print("Creating dataset object...")
    dataset = ImageReconstructionDataset(
        compressed_base_path=compressed_path,
        uncompressed_base_path=uncompressed_path,
        target_size=(256, 256),  # Choose appropriate size
        transform=None
    )
    print(f"✓ Dataset created with {len(dataset)} samples")

    # Create data loader
    batch_size = 16
    print(f"Initializing DataLoader with batch size {batch_size}...")

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    print("✓ DataLoader initialized")

    # Prepare all the data outside the training loop
    print("Preparing data batches...")
    prepared_data = list(data_loader)
    print(f"✓ Data prepared: {len(prepared_data)} batches")

    print("Initializing models and moving to CUDA...")
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    print("✓ Models created and moved to CUDA")

    print("Setting up optimizers...")
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    print("✓ Optimizers configured")

    print("\n" + "=" * 50)
    print("BEGINNING TRAINING")
    print("=" * 50)

    total_epochs = 20
    # Training loop
    for epoch in range(total_epochs):
        print(f"\n{'=' * 20} EPOCH {epoch + 1}/{total_epochs} {'=' * 20}")
        epoch_start_time = time.time()

        for i, data in enumerate(prepared_data):
            batch_start_time = time.time()
            print(f"\n--- Batch {i + 1}/{len(prepared_data)} ---")

            # Unpack the tuple returned by the dataset
            print("  • Unpacking data batch...")
            input_dict, target_tensor = data

            # Get data from input_dict and move to cuda
            compressed_images = input_dict['compressed_image'].cuda()
            camera_labels = input_dict['camera_idx'].cuda()
            real_images = target_tensor.cuda()
            print(
                f"  • Data moved to CUDA (compressed: {compressed_images.shape}, camera labels: {camera_labels.shape})")

            # Train discriminator
            print("  • Training discriminator...")
            disc_losses = train_discriminator(discriminator, generator, real_images, compressed_images, camera_labels)
            disc_loss_sum = sum(disc_losses.values())
            print(f"  ✓ Discriminator training complete - Loss: {disc_loss_sum:.4f}")

            # Train generator
            print("  • Training generator...")
            gen_losses = train_generator(generator, discriminator, compressed_images, camera_labels)
            gen_loss_sum = sum(gen_losses.values())
            print(f"  ✓ Generator training complete - Loss: {gen_loss_sum:.4f}")

            batch_time = time.time() - batch_start_time

            # Print progress
            if i % 10 == 0:
                print(f"\nBatch {i + 1}/{len(prepared_data)} Summary:")
                print(f"  - D_loss={disc_loss_sum:.4f}")
                print(f"    - Details: {', '.join([f'{k}={v:.4f}' for k, v in disc_losses.items()])}")
                print(f"  - G_loss={gen_loss_sum:.4f}")
                print(f"    - Details: {', '.join([f'{k}={v:.4f}' for k, v in gen_losses.items()])}")
                print(f"  - Batch processing time: {batch_time:.2f} seconds")

        epoch_time = time.time() - epoch_start_time
        print(f"\n{'=' * 20} EPOCH {epoch + 1} COMPLETE {'=' * 20}")
        print(f"Time taken: {epoch_time:.2f} seconds")
