import rawpy



# Load and process a DNG file
def load_dng(image_path):
    """ Load a DNG file and convert it to an RGB NumPy array"""
    with rawpy.imread(image_path) as raw:
        rgb_image = raw.postprocess()
    return rgb_image  # RGB NumPy array



# TODO: extract SPN function
# TODO: function that parses image, output of gray_scale and spn
# TODO: Function that call previous on all images
