import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)

class Generator(nn.Module):
    def __init__(self, input_channels=3, spn_channels=1, output_channels=1, num_compression_levels=100):
        super(Generator, self).__init__()
        
        # Compression level embedding
        self.compression_embedding = nn.Embedding(num_compression_levels, 256)
        
        # Encoder for compressed image
        self.enc1_img = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.enc2_img = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc3_img = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc4_img = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.enc5_img = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        
        # Encoder for compressed SPN (single channel)
        self.enc1_spn = nn.Conv2d(spn_channels, 64, kernel_size=4, stride=2, padding=1)
        self.enc2_spn = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc3_spn = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc4_spn = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.enc5_spn = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        
        # Attention blocks with correct channel sizes
        self.attention1 = AttentionBlock(1024)  # After concatenating e5_img and e5_spn (512 + 512)
        self.attention2 = AttentionBlock(1536)  # After concatenating d1, e4_img, e4_spn (512 + 512 + 512)
        self.attention3 = AttentionBlock(768)   # After concatenating d2, e3_img, e3_spn (256 + 256 + 256)
        self.attention4 = AttentionBlock(384)   # After concatenating d3, e2_img, e2_spn (128 + 128 + 128)
        
        # Decoder with skip connections
        self.dec1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(1536, 256, kernel_size=4, stride=2, padding=1)  # Input: 1536 (512*3)
        self.dec3 = nn.ConvTranspose2d(768, 128, kernel_size=4, stride=2, padding=1)   # Input: 768 (256*3)
        self.dec4 = nn.ConvTranspose2d(384, 64, kernel_size=4, stride=2, padding=1)    # Input: 384 (128*3)
        self.dec5 = nn.ConvTranspose2d(192, output_channels, kernel_size=4, stride=2, padding=1)  # Input: 192 (64*3)
        
        # Batch normalization layers
        self.bn_enc2_img = nn.BatchNorm2d(128)
        self.bn_enc3_img = nn.BatchNorm2d(256)
        self.bn_enc4_img = nn.BatchNorm2d(512)
        self.bn_enc5_img = nn.BatchNorm2d(512)
        
        self.bn_enc2_spn = nn.BatchNorm2d(128)
        self.bn_enc3_spn = nn.BatchNorm2d(256)
        self.bn_enc4_spn = nn.BatchNorm2d(512)
        self.bn_enc5_spn = nn.BatchNorm2d(512)
        
        self.bn_dec1 = nn.BatchNorm2d(512)
        self.bn_dec2 = nn.BatchNorm2d(256)
        self.bn_dec3 = nn.BatchNorm2d(128)
        self.bn_dec4 = nn.BatchNorm2d(64)
        
        # Compression level conditioning
        self.compression_conv = nn.Conv2d(256, 512, 1)
        
    def forward(self, compressed_img, compressed_spn, compression_level):
        # Get compression level embedding
        comp_emb = self.compression_embedding(compression_level)
        comp_emb = comp_emb.view(-1, 256, 1, 1)
        comp_emb = self.compression_conv(comp_emb)
        
        # Encode compressed image
        e1_img = F.leaky_relu(self.enc1_img(compressed_img), 0.2)
        e2_img = F.leaky_relu(self.bn_enc2_img(self.enc2_img(e1_img)), 0.2)
        e3_img = F.leaky_relu(self.bn_enc3_img(self.enc3_img(e2_img)), 0.2)
        e4_img = F.leaky_relu(self.bn_enc4_img(self.enc4_img(e3_img)), 0.2)
        e5_img = F.leaky_relu(self.bn_enc5_img(self.enc5_img(e4_img)), 0.2)
        
        # Encode compressed SPN (single channel)
        e1_spn = F.leaky_relu(self.enc1_spn(compressed_spn), 0.2)
        e2_spn = F.leaky_relu(self.bn_enc2_spn(self.enc2_spn(e1_spn)), 0.2)
        e3_spn = F.leaky_relu(self.bn_enc3_spn(self.enc3_spn(e2_spn)), 0.2)
        e4_spn = F.leaky_relu(self.bn_enc4_spn(self.enc4_spn(e3_spn)), 0.2)
        e5_spn = F.leaky_relu(self.bn_enc5_spn(self.enc5_spn(e4_spn)), 0.2)
        
        # Combine features with compression level
        combined = torch.cat([e5_img, e5_spn], dim=1)  # 1024 channels
        combined = self.attention1(combined)
        combined = combined + comp_emb.repeat(1, 2, 1, 1)  # Repeat comp_emb to match combined channels
        
        # Decoder with skip connections and attention
        d1 = F.relu(self.bn_dec1(self.dec1(combined)))
        d1 = torch.cat([d1, e4_img, e4_spn], 1)  # 1536 channels
        d1 = self.attention2(d1)
        
        d2 = F.relu(self.bn_dec2(self.dec2(d1)))
        d2 = torch.cat([d2, e3_img, e3_spn], 1)  # 768 channels
        d2 = self.attention3(d2)
        
        d3 = F.relu(self.bn_dec3(self.dec3(d2)))
        d3 = torch.cat([d3, e2_img, e2_spn], 1)  # 384 channels
        d3 = self.attention4(d3)
        
        d4 = F.relu(self.bn_dec4(self.dec4(d3)))
        d4 = torch.cat([d4, e1_img, e1_spn], 1)  # 192 channels
        
        # Output restored SPN (single channel)
        restored_spn = torch.tanh(self.dec5(d4))
        
        return restored_spn 