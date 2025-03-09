import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Activation, \
    Concatenate, Lambda


def build_generator(input_shape=(256, 256, 3), spn_shape=(256, 256, 1), num_compression_levels=3):
    """ U-Net Generator for restoring SPN """

    # Inputs: Compressed image, SPN from compressed image, Compression level
    comp_image = Input(shape=input_shape, name="compressed_image")
    comp_spn = Input(shape=spn_shape, name="compressed_spn")
    compression_label = Input(shape=(num_compression_levels,), name="compression_label")

    # Incorporate condition: reshape & tile label to image shape
    cond = Lambda(lambda c: tf.reshape(c, (-1, 1, 1, num_compression_levels)))(compression_label)
    cond = Lambda(lambda c: tf.tile(c, (1, input_shape[0], input_shape[1], 1)))(cond)

    # Concatenate inputs along channels
    x = Concatenate(axis=-1)([comp_image, comp_spn, cond])  # Shape: (H, W, 3 + 1 + num_compression_levels)

    # Encoder (Downsampling)
    d1 = Conv2D(64, kernel_size=4, strides=2, padding='same')(x)
    d1 = LeakyReLU(0.2)(d1)
    d2 = Conv2D(128, kernel_size=4, strides=2, padding='same')(d1)
    d2 = BatchNormalization()(d2);
    d2 = LeakyReLU(0.2)(d2)
    d3 = Conv2D(256, kernel_size=4, strides=2, padding='same')(d2)
    d3 = BatchNormalization()(d3);
    d3 = LeakyReLU(0.2)(d3)

    # Decoder (Upsampling)
    u2 = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(d3)
    u2 = BatchNormalization()(u2);
    u2 = Activation('relu')(u2)
    u2 = Concatenate()([u2, d2])
    u1 = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(u2)
    u1 = BatchNormalization()(u1);
    u1 = Activation('relu')(u1)
    u1 = Concatenate()([u1, d1])

    # Output: Predicted SPN residual
    output_spn = Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh')(u1)

    return tf.keras.Model([comp_image, comp_spn, compression_label], output_spn, name="Generator")


def build_discriminator(input_shape=(256, 256, 3), spn_shape=(256, 256, 1), num_compression_levels=3):
    """ PatchGAN Discriminator """

    comp_image = Input(shape=input_shape, name="compressed_image")
    target_spn = Input(shape=spn_shape, name="target_spn")  # SPN residual
    compression_label = Input(shape=(num_compression_levels,), name="compression_label")

    # Incorporate condition as additional input channel
    cond = Lambda(lambda c: tf.reshape(c, (-1, 1, 1, num_compression_levels)))(compression_label)
    cond = Lambda(lambda c: tf.tile(c, (1, input_shape[0], input_shape[1], 1)))(cond)

    x = Concatenate(axis=-1)([comp_image, target_spn, cond])

    f = 64
    x = Conv2D(f, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(f * 2, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x);
    x = LeakyReLU(0.2)(x)
    x = Conv2D(f * 4, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x);
    x = LeakyReLU(0.2)(x)
    x = Conv2D(f * 8, kernel_size=4, strides=1, padding='same')(x)
    x = BatchNormalization()(x);
    x = LeakyReLU(0.2)(x)

    patch_out = Conv2D(1, kernel_size=4, strides=1, padding='same')(x)

    return tf.keras.Model([comp_image, target_spn, compression_label], patch_out, name="Discriminator")
