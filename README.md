# 🎨 GAN-based Sensor Pattern Noise Restoration

> A powerful tool for restoring sensor pattern noise in compressed images using Generative Adversarial Networks

## 📁 Project Structure

The project follows a specific file organization structure to manage different types of images and their processing stages. All paths are configured through environment variables in the `.env` file.

### 📂 Directory Structure

```bash
EXTERNAL_DRIVE/
├── Images/
│   ├── Original/           # Original RAW files (DNG and RW2)
│   │   ├── Camera_Model_1/
│   │   │   ├── image1.DNG
│   │   │   ├── image2.DNG
│   │   │   └── ...
│   │   ├── Camera_Model_2/
│   │   │   ├── image1.RW2
│   │   │   ├── image2.RW2
│   │   │   └── ...
│   │   └── ...
│   │
│   ├── Compressed/         # Compressed JPEG files
│   │   ├── Camera_Model_1/
│   │   │   ├── 90/         # 90% quality JPEGs
│   │   │   ├── 60/         # 60% quality JPEGs
│   │   │   └── 30/         # 30% quality JPEGs
│   │   └── ...
│   │
│   └── SPN/                # Sensor Pattern Noise files
│       ├── Camera_Model_1/
│       │   ├── original/
│       │   │   └── camera_SPN.png    # SPN from one randomly selected RAW image
│       │   └── compressed/
│       │       ├── 30/     # SPNs from 30% quality JPEGs
│       │       ├── 60/     # SPNs from 60% quality JPEGs
│       │       └── 90/     # SPNs from 90% quality JPEGs
│       └── ...
```

### ✨ Key Features

- **📷 Camera Model Organization**: Each camera model has its own directory
- **🖼️ RAW Format Support**: Supports DNG and RW2 file formats
- **📊 Compression Levels**: JPEG compression at 90%, 60%, and 30% quality
- **🔍 SPN Extraction**: 
  - Single RAW image SPN extraction per camera
  - Individual SPN extraction from compressed images
- **⚙️ Environment Configuration**: Easy path management through `.env` file
- **⏱️ Progress Tracking**: Visual progress bars for processing status

## 🛠️ Setup

### Environment Configuration

Create a `.env` file based on the provided template (`.env.example`):

```env
# External drive path
EXTERNAL_DRIVE=/path/to/your/external/drive/

# Image directories (relative to EXTERNAL_DRIVE)
ORIGINAL_IMAGES_DIR=Images/Original
COMPRESSED_IMAGES_DIR=Images/Compressed
SPN_IMAGES_DIR=Images/SPN

# JPEG quality levels
QUALITY_LEVELS=90,60,30
```

### 🚀 Quick Start

1. Copy `.env.example` to `.env`
2. Update paths in `.env` to match your system
3. Connect your external drive
4. Run the pipeline:
   ```bash
   python pipeline.py
   ```

## 🔄 Processing Pipeline

The pipeline consists of two main steps:

### 1. Image Compression
- Converts RAW images to JPEG format
- Creates multiple quality versions (30%, 60%, 90%)
- Organizes by camera model and quality
- Skips already processed images
- Shows progress bars for processing status

### 2. SPN Extraction
- Extracts Sensor Pattern Noise from:
  - One randomly selected RAW image per camera
  - All compressed images at each quality level
- Creates camera SPN from single RAW image
- Generates individual SPNs from compressed images
- Organizes by camera model and quality
- Shows progress bars for processing status

## 🤖 Machine Learning Dataset

### Input Features:
- **Compressed Image** (RGB, 3 channels)
- **Compressed Image SPN** (Grayscale, 1 channel)
- **Quality Level** (Scalar)

### Target:
- **Camera SPN** (Grayscale, 1 channel)

### Usage Example:
```python
dataset = SPNRestorationDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    inputs, targets = batch
    # Process your data...
```

## 🚧 Current Status

The GAN implementation is being redesigned to match the new dataset structure. The new implementation will:

- **Input**:
  - Compressed JPEG image (3 channels)
  - Compressed image's SPN (1 channel)
  - JPEG quality level
- **Output**:
  - Restored SPN (1 channel)

## 🔮 Planned Improvements

1. **Hourglass Structure** 🏗️
   - Improved skip connections
   - Enhanced feature preservation
   - Better handling of compression artifacts

2. **Frequency-based Loss Functions** 📊
   - Low Frequency (LF) loss
   - High Frequency (HF) loss with VGG-16
   - Combined loss optimization

3. **Blocking Artifact Removal** 🧹
   - Enhanced JPEG artifact handling
   - Better image quality preservation
   - Improved SPN pattern restoration

## 📚 References

[1] Si, J., & Kim, S. (2024). Restoration of the JPEG Maximum Lossy Compressed Face Images with Hourglass Block-GAN. Computers, Materials & Continua, 78(3), 2893-2908. https://doi.org/10.32604/cmc.2023.046081

## 📋 Requirements

- Python 3.x
- External storage device
- Required Python packages (see `requirements.txt`)