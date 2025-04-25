import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyAnalysis(nn.Module):
    def __init__(self):
        super(FrequencyAnalysis, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Initialize tensors for magnitude and phase
        fft_mag = torch.zeros(batch_size, channels, height, width).to(x.device)
        fft_phase = torch.zeros(batch_size, channels, height, width).to(x.device)
        
        # Compute FFT for each channel separately
        for c in range(channels):
            fft = torch.fft.fft2(x[:, c])
            fft_mag[:, c] = torch.abs(fft)
            fft_phase[:, c] = torch.angle(fft)
        
        # Average across channels to get a single magnitude and phase
        fft_mag = torch.mean(fft_mag, dim=1)
        fft_phase = torch.mean(fft_phase, dim=1)
        
        # Stack magnitude and phase
        fft_features = torch.stack([fft_mag, fft_phase], dim=1)  # Shape: [batch_size, 2, height, width]
        
        # Process frequency features with strided convolutions to match spatial path
        x = F.leaky_relu(self.bn1(self.conv1(fft_features)), 0.2)  # 1/2 spatial size
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)            # 1/4 spatial size
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)            # 1/8 spatial size
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)            # 1/16 spatial size
        
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, num_cameras=10):
        super(Discriminator, self).__init__()
        self.num_cameras = num_cameras
        
        # Spatial feature extraction
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)  # 1/2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)            # 1/4
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)          # 1/8
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)          # 1/16
        
        # Frequency domain analysis
        self.freq_analysis = FrequencyAnalysis()
        
        # Feature fusion (512 from spatial + 256 from freq = 768)
        self.fusion_conv = nn.Conv2d(768, 512, kernel_size=1)
        
        # Real/Fake classification head
        self.authenticity_head = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        
        # Camera identification head
        self.camera_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=4, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_cameras)
        )
        
        # SPN pattern analysis head
        self.spn_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=4, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
    def forward(self, image):
        # Spatial feature extraction (with size annotations)
        x = F.leaky_relu(self.conv1(image), 0.2)                    # 1/2
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)             # 1/4
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)             # 1/8
        spatial_features = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)  # 1/16
        
        # Frequency domain analysis (now matches spatial dimensions)
        freq_features = self.freq_analysis(image)  # Will be 1/16 size
        
        # Feature fusion (both inputs now have matching spatial dimensions)
        combined_features = torch.cat([spatial_features, freq_features], dim=1)
        features = self.fusion_conv(combined_features)
        
        # Get predictions (keeping dimensions consistent)
        authenticity = self.authenticity_head(features)
        camera_logits = self.camera_head(features)
        spn_pattern = self.spn_head(features)
        
        # Ensure consistent shapes: [batch_size, 1] for binary outputs
        return (
            authenticity.view(-1, 1),  # [batch_size, 1]
            camera_logits,             # [batch_size, num_cameras]
            spn_pattern.view(-1, 1)    # [batch_size, 1]
        ) 