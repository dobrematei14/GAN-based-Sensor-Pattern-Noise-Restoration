from Image_loading import retrieveDNG

import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet


def convert_to_float(image):
    # takes an image as input
    # returns the image as a float
    return image.astype(np.float32) / 255.0


def wavelet_denoise(image):
    # takes an image as input
    # denoises it using wavelet denoising
    # and returns the denoised image
    image = convert_to_float(image)
    denoised = denoise_wavelet(
        image,
        multichannel=True,
        convert2ycbcr=True,
        method='BayesShrink',
        mode='soft'
    )
    return image, denoised

def SPN_extraction(image):
    # takes an image as input
    # extracts the SPN from it
    # and returns the SPN
    image, denoised = wavelet_denoise(image)
    SPN = image - denoised
    spn_normalized = (SPN - np.min(SPN)) / (np.max(SPN) - np.min(SPN))

    return spn_normalized


def previewSPN(dng_path):
    # takes a path to a DNG file
    # extracts the SPN from it
    # and displays it
    dng = retrieveDNG(dng_path)
    SPN = SPN_extraction(dng)
    plt.imshow(SPN)
    plt.show()


previewSPN("Images/Original Format/1.DNG")