import torch
import torch.nn as nn
import torch.nn.functional as F

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