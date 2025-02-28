import rawpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import imageio

def previewDNG(dng_path):
# takes a path to a DNG file
# converts it to an RGB image
# and displays it
    dng = rawpy.imread(dng_path)
    rgb = dng.postprocess()
    plt.imshow(rgb)
    plt.show()

def retrieveDNG(dng_path):
    # takes a path to a DNG file
    # converts it to an RGB image
    # and returns it
    dng = rawpy.imread(dng_path)
    rgb = dng.postprocess()
    return rgb

def compressDNG(dng_path, save_dir):
    # takes a path to a DNG file
    # convert to a PIL image
    # and save it as JPEG with the same name
    dng = rawpy.imread(dng_path)
    rgb = dng.postprocess()
    base_name = os.path.basename(dng_path)
    name, _ = os.path.splitext(base_name)
    save_path = os.path.join(save_dir, f"{name}.jpeg")
    imageio.imwrite(save_path, rgb)

def compressFolderDNG(folder_path, save_dir):
    # takes a path to a folder containing DNG files
    # converts them to JPEG images
    # and saves them in save_dir
    for file in os.listdir(folder_path):
        if file.endswith(".DNG"):
            file_path = os.path.join(folder_path, file)
            compressDNG(file_path, save_dir)


compressFolderDNG("Images/Original Format/", "Images/Compressed/")