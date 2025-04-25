import pywt
import numpy as np
import cv2
import os
from skimage.restoration import denoise_wavelet


def save_spn(img_path, out_path, wavelet='db1', level=1, cam=None, fmt='png'):
    """Extract SPN from an image and save it as a PNG file."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    spn = extract_spn(img, wavelet, level)
    spn_norm = cv2.normalize(spn, None, 0, 255, cv2.NORM_MINMAX)
    spn_norm = np.uint8(spn_norm)
    
    out_name = f"{cam}_SPN.{fmt}" if cam else f"{os.path.splitext(os.path.basename(img_path))[0]}_SPN.{fmt}"
    cv2.imwrite(os.path.join(out_path, out_name), spn_norm)


def extract_spn(img, wavelet='db1', level=1):
    """Extract SPN using wavelet decomposition."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coeffs = pywt.wavedec2(img, wavelet, level=level)
    coeffs = [coeffs[0] * 0] + list(coeffs[1:])
    spn = pywt.waverec2(coeffs, wavelet)
    spn = (spn - np.mean(spn)) / np.std(spn)

    return spn


def extract_spn_denoise(img, wavelet='db8', is_color=False):
    """Extract SPN using scikit-image's denoise_wavelet function."""
    if len(img.shape) == 3 and not is_color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_float = img.astype(float)
    denoised = denoise_wavelet(
        img_float,
        method='BayesShrink',
        mode='soft',
        wavelet=wavelet,
        rescale_sigma=True,
        channel_axis=-1 if is_color else None,
        convert2ycbcr=True if is_color else False
    )

    noise = img_float - denoised
    noise = (noise - np.mean(noise)) / np.std(noise)

    return noise
