from SPN.SPN_extraction_methods import extract_spn_wavelet
import cv2
import numpy as np


def calculate_pce(noise1, noise2):
    """
    Calculate the Peak to Correlation Energy (PCE) between two noise patterns.
    
    PCE is a measure used to determine the similarity between two sensor pattern noise (SPN) signals.
    Higher PCE values indicate a stronger match between the noise patterns.
    
    Args:
        noise1 (numpy.ndarray): First noise pattern
        noise2 (numpy.ndarray): Second noise pattern
        
    Returns:
        float: PCE value representing the similarity between the two noise patterns
    """
    # Compute normalized cross-correlation
    correlation = cv2.matchTemplate(noise1, noise2, cv2.TM_CCORR_NORMED)
    
    # Find the peak
    _, max_val, _, max_loc = cv2.minMaxLoc(correlation)
    
    # Define a small region around the peak to exclude (typically 11x11)
    peak_radius = 5
    peak_region = correlation[max_loc[1]-peak_radius:max_loc[1]+peak_radius+1, 
                            max_loc[0]-peak_radius:max_loc[0]+peak_radius+1]
                            
    # Calculate energy (excluding the peak region)
    mask = np.ones_like(correlation)
    mask[max_loc[1]-peak_radius:max_loc[1]+peak_radius+1, 
         max_loc[0]-peak_radius:max_loc[0]+peak_radius+1] = 0
    
    # Calculate the energy of the correlation plane excluding the peak
    energy = np.sum(correlation**2 * mask) / (np.sum(mask))
    
    # Calculate PCE
    pce = max_val**2 / energy
    
    return pce


def noise_correlation(noise1, noise2):
    """
    Calculate the correlation coefficient between two noise patterns after extracting their SPNs.
    
    This function extracts the sensor pattern noise from both input images and then
    calculates the correlation coefficient between them.
    
    Args:
        noise1 (numpy.ndarray): First image to extract noise from
        noise2 (numpy.ndarray): Second image to extract noise from
        
    Returns:
        float: Correlation coefficient between the two noise patterns (-1 to 1)
    """
    # Extract sensor pattern noise from both images
    noise1 = extract_spn_wavelet(noise1)
    noise2 = extract_spn_wavelet(noise2)

    # Calculate correlation coefficient between flattened noise patterns
    correlation = np.corrcoef(noise1.flatten(), noise2.flatten())[0, 1]
    return correlation


def camera_identification_test(test_image, camera_spn, threshold=50):
    """
    Test if an image was taken by a specific camera by comparing its SPN with the camera's reference SPN.
    
    This function extracts the sensor pattern noise from a test image and compares it with
    a reference camera SPN to determine if the image was likely taken by that camera.
    
    Args:
        test_image (numpy.ndarray): The test image to identify
        camera_spn (numpy.ndarray): The reference SPN of the camera
        threshold (float, optional): PCE threshold for positive identification. Defaults to 50.
        
    Returns:
        dict: Results containing:
            - match (bool): True if the image matches the camera
            - pce (float): Peak to Correlation Energy value
            - correlation (float): Correlation coefficient between SPNs
    """
    # Extract SPN from test image
    test_spn = extract_spn_wavelet(test_image)
    
    # Calculate PCE between test SPN and camera reference SPN
    pce_value = calculate_pce(test_spn, camera_spn)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(test_spn.flatten(), camera_spn.flatten())[0, 1]
    
    # Determine if there's a match based on PCE threshold
    is_match = pce_value > threshold
    
    return {
        'match': is_match,
        'pce': pce_value,
        'correlation': correlation
    }




# create a function that takes an image, removes its SPN, reconstructs it using the model,
# and then compares them