# GAN-based Sensor Pattern Noise Restoration

## Project Structure

The project follows a specific file organization structure to manage different types of images and their processing stages. All paths are configured through environment variables in the `.env` file.

### Directory Structure

```
EXTERNAL_DRIVE/
├── Images/
│   ├── Original/           # Original DNG files
│   │   ├── Camera_Model_1/
│   │   │   ├── image1.DNG
│   │   │   ├── image2.DNG
│   │   │   └── ...
│   │   ├── Camera_Model_2/
│   │   │   ├── image1.DNG
│   │   │   ├── image2.DNG
│   │   │   └── ...
│   │   └── ...
│   │
│   ├── Compressed/         # Compressed JPEG files
│   │   ├── Camera_Model_1/
│   │   │   ├── 90/         # 90% quality JPEGs
│   │   │   │   ├── image1.jpg
│   │   │   │   ├── image2.jpg
│   │   │   │   └── ...
│   │   │   ├── 60/         # 60% quality JPEGs
│   │   │   │   ├── image1.jpg
│   │   │   │   ├── image2.jpg
│   │   │   │   └── ...
│   │   │   └── 30/         # 30% quality JPEGs
│   │   │       ├── image1.jpg
│   │   │       ├── image2.jpg
│   │   │       └── ...
│   │   ├── Camera_Model_2/
│   │   │   └── ...         # Same structure as above
│   │   └── ...
│   │
│   └── SPN/                # Sensor Pattern Noise files
│       ├── Camera_Model_1/
│       │   ├── SPN/        # SPN extraction results
│       │   │   ├── spn1.png
│       ├── Camera_Model_2/
│       │   └── ...         # Same structure as above
│       └── ...
```

### Key Features

1. **Camera Model Organization**: Each camera model has its own directory, maintaining separation between different devices.
2. **Compression Levels**: JPEG compression is performed at multiple quality levels (90%, 60%, 30%) for each image.
3. **SPN Extraction**: Sensor Pattern Noise is extracted and stored separately for each camera model.
4. **Environment Configuration**: All paths are managed through the `.env` file, making it easy to adapt to different systems.

### Environment Configuration

The project uses a `.env` file for configuration. A template `.env.example` is provided with the following structure:

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

To get started:
1. Copy `.env.example` to `.env`
2. Update the paths in `.env` to match your system
3. Ensure your external drive is connected and accessible

## Processing Pipeline

1. **Original Images**: Start with DNG files in the Original directory
2. **Compression**: Generate compressed JPEG versions at different quality levels
3. **SPN Extraction**: Extract Sensor Pattern Noise from both original and compressed images
4. **Dataset Generation**: Create training pairs from compressed images and their SPN
5. **Model Training**: Train the GAN model using the generated dataset

## Current Implementation Status

The current GAN implementation serves as a placeholder and basic framework for the project. It includes:

- Basic generator and discriminator architectures
- Simple training loop with basic loss functions
- Environment variable-based configuration
- Basic logging and progress tracking

This implementation provides a foundation for the following planned improvements.

## Planned Improvements

Based on recent research in image restoration using GANs, particularly the Hourglass Block-GAN approach [1], several significant improvements are planned:

1. **Hourglass Structure**: 
   - Implementing a new hourglass architecture that preserves deep layer characteristics
   - Better handling of high compression artifacts through improved skip connections
   - Enhanced feature preservation in the encoder-decoder structure

2. **Frequency-based Loss Functions**:
   - Low Frequency (LF) loss for handling smooth regions
   - High Frequency (HF) loss using pretrained VGG-16 for detailed features
   - Combined loss functions to improve restoration of both fine details and overall structure

3. **Blocking Artifact Removal**:
   - Enhanced handling of JPEG compression artifacts
   - Better preservation of image identity and quality at high compression rates
   - Improved SPN pattern preservation during restoration

These improvements will be particularly valuable for our SPN restoration task, as they address similar challenges in image quality restoration and artifact removal.

[1] Si, J., & Kim, S. (2024). Restoration of the JPEG Maximum Lossy Compressed Face Images with Hourglass Block-GAN. Computers, Materials & Continua, 78(3), 2893-2908. https://doi.org/10.32604/cmc.2023.046081

## Requirements

- Python 3.x
- External storage device (for image storage)
- Required Python packages (see requirements.txt)