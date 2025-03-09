import tensorflow as tf
from dataset_generation import get_dataset
from cGAN_architecture import build_generator, build_discriminator
import numpy as np

# Load dataset
batch_size = 8
dataset = get_dataset(batch_size=batch_size)

# Build models
generator = build_generator()
discriminator = build_discriminator()

# Optimizers
gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Loss functions
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Custom SPN loss (MSE on noise residuals)
def spn_loss(real_spn, fake_spn):
    return tf.reduce_mean(tf.square(real_spn - fake_spn))


# Training loop
def train_cgan(epochs=50):
    for epoch in range(epochs):
        gen_losses = []
        disc_losses = []

        for comp_img, orig_img, comp_spn, orig_spn, compression_label, _ in dataset:
            # Discriminator training
            with tf.GradientTape() as disc_tape:
                fake_spn = generator([comp_img, comp_spn, compression_label], training=True)

                real_output = discriminator([comp_img, orig_spn, compression_label], training=True)
                fake_output = discriminator([comp_img, fake_spn, compression_label], training=True)

                real_loss = bce(tf.ones_like(real_output), real_output)
                fake_loss = bce(tf.zeros_like(fake_output), fake_output)
                disc_loss = real_loss + fake_loss
                disc_losses.append(disc_loss)

            disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

            # Generator training
            with tf.GradientTape() as gen_tape:
                fake_spn = generator([comp_img, comp_spn, compression_label], training=True)
                fake_output = discriminator([comp_img, fake_spn, compression_label], training=False)

                adv_loss = bce(tf.ones_like(fake_output), fake_output)
                spn_recon_loss = spn_loss(orig_spn, fake_spn)
                gen_loss = adv_loss + 100 * spn_recon_loss
                gen_losses.append(gen_loss)

            gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

        # Calculate average losses for the epoch
        avg_gen_loss = tf.reduce_mean(gen_losses).numpy()
        avg_disc_loss = tf.reduce_mean(disc_losses).numpy()
        print(f"Epoch {epoch + 1}/{epochs}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            generator.save(f'generator_epoch_{epoch + 1}.h5')
            discriminator.save(f'discriminator_epoch_{epoch + 1}.h5')
            print("Models saved.")

if __name__ == "__main__":
    train_cgan()