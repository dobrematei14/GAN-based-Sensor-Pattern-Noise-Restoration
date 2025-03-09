import tensorflow as tf
import numpy as np
import cv2
import os
from dataset_generation import get_dataset
from cGAN.cGAN_architecture import build_generator
from SPN.SPN_extraction import extract_spn

# Load trained generator
generator = tf.keras.models.load_model('generator_epoch_50.h5')

# Load dataset
dataset = get_dataset(batch_size=1)


# Evaluation Metrics
def compute_psnr(original, restored):
    return tf.image.psnr(original, restored, max_val=1.0).numpy()


def compute_ssim(original, restored):
    return tf.image.ssim(original, restored, max_val=1.0).numpy()


def compute_correlation(original_spn, restored_spn):
    original_spn = original_spn.flatten()
    restored_spn = restored_spn.flatten()
    return np.corrcoef(original_spn, restored_spn)[0, 1]


# Prepare log file
log_file = "evaluation_results.txt"
if os.path.exists(log_file):
    os.remove(log_file)

# Evaluate on test samples and log results
with open(log_file, "w") as f:
    f.write("PSNR, SSIM, Correlation\n")

    for i, (comp_img, orig_img, comp_spn, orig_spn, compression_label, _) in enumerate(dataset.take(100)):
        restored_spn = generator([comp_img, comp_spn, compression_label], training=False)

        # Compute metrics
        psnr = compute_psnr(orig_spn, restored_spn)
        ssim = compute_ssim(orig_spn, restored_spn)
        correlation = compute_correlation(orig_spn.numpy(), restored_spn.numpy())

        log_entry = f"{psnr:.2f}, {ssim:.4f}, {correlation:.4f}\n"
        f.write(log_entry)

        print(f"Sample {i + 1}: PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, Correlation: {correlation:.4f}")

print(f"Evaluation completed. Results saved to {log_file}")
