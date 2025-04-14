import torch
import torch.nn as nn
import torch.nn.functional as F

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