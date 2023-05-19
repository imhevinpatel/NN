# Patel, Hevin Dharmeshbhai
# 1002_036_919
# 2023_04_30
# Assignment_05_01

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import layers

def plot_images(generated_images, n_rows=1, n_cols=10):
    """
    Plot the images in a 1x10 grid
    :param generated_images:
    :return:
    """
    f, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    ax = ax.flatten()
    for i in range(n_rows*n_cols):
        ax[i].imshow(generated_images[i, :, :], cmap='gray')
        ax[i].axis('off')
    return f, ax

class GenerateSamplesCallback(tf.keras.callbacks.Callback):
    """
    Callback to generate images from the generator model at the end of each epoch
    Uses the same noise vector to generate images at each epoch, so that the images can be compared over time
    """
    def __init__(self, generator, noise):
        self.generator = generator
        self.noise = noise

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists("generated_images"):
            os.mkdir("generated_images")
        generated_images = self.generator(self.noise, training=False)
        generated_images = generated_images.numpy()
        generated_images = generated_images*127.5 + 127.5
        generated_images = generated_images.reshape((10, 28, 28))
        # plot images using matplotlib
        plot_images(generated_images)
        plt.savefig(os.path.join("generated_images", f"generated_images_{epoch}.png"))
        # close the plot to free up memory
        plt.close()

def build_discriminator():
    # sequential model
    model = tf.keras.models.Sequential()

    # convolutional layer
    model.add(layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
    # LeakyReLU activation function to introduce non-linearity
    model.add(layers.LeakyReLU())
    # dropout layer with a rate of 0.3 to prevent overfitting
    model.add(layers.Dropout(0.3))
    # Adding another convolutional layer
    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    # Adding another LeakyReLU activation function
    model.add(layers.LeakyReLU())
    # Adding another dropout layer
    model.add(layers.Dropout(0.3))
    # Flatten the output from the previous layer
    model.add(layers.Flatten())
    # Adding a dense layer with 1 unit to perform classification
    model.add(layers.Dense(1))
    return model

def build_generator():
    # Create a sequential model
    model = tf.keras.models.Sequential()
    # Adding a dense layer
    # Use_bias is set to False to ensure centering in the BatchNormalization layer
    model.add(layers.Dense(units=7*7*8, use_bias=False, input_shape=(100,)))
    # Adding a BatchNormalization layer to normalize the activations and improve training stability
    model.add(layers.BatchNormalization())
    # Adding a LeakyReLU activation function to introduce non-linearity
    model.add(layers.LeakyReLU())
    # Reshape the output from the previous layer
    model.add(layers.Reshape(target_shape=(7, 7, 8)))
    # Adding a transposed convolutional layer
    model.add(layers.Conv2DTranspose(filters=8, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False))
    # Adding another BatchNormalization layer
    model.add(layers.BatchNormalization())
    # Adding another LeakyReLU activation function
    model.add(layers.LeakyReLU())
    # Adding another transposed convolutional layer
    model.add(layers.Conv2DTranspose(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False))
    # Adding another BatchNormalization layer
    model.add(layers.BatchNormalization())
    # Adding another LeakyReLU activation function
    model.add(layers.LeakyReLU())
    # Adding the final transposed convolutional layer
    # activation function to generate the final output image
    model.add(layers.Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model


class DCGAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        noise = tf.random.uniform([batch_size, 100])

        # Gradient tapes to track the operations and calculate gradients
        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            # Generate fake images
            images = self.generator(noise, training=True)

            # Get discriminator predictions for real and fake images
            real = self.discriminator(data, training=True)
            fake = self.discriminator(images, training=True)

            # Generator loss
            g_loss = self.loss_fn(tf.ones_like(fake), fake)

            # Discriminator loss
            real_labels = tf.ones_like(real)
            fake_labels = tf.zeros_like(fake)
            concat_output = tf.concat([real, fake], axis=0)
            concat_labels = tf.concat([real_labels, fake_labels], axis=0)
            d_loss = self.loss_fn(concat_labels, concat_output)

        # Compute gradients
        d_gradient = discriminator_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_gradient = generator_tape.gradient(g_loss, self.generator.trainable_variables)

        # Update discriminator and generator weights using the gradients
        self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))

        return {'d_loss': d_loss, 'g_loss': g_loss}


def train_dcgan_mnist():
    tf.keras.utils.set_random_seed(5368)
    # load mnist
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # the images are in the range [0, 255], we need to rescale them to [-1, 1]
    x_train = (x_train - 127.5) / 127.5
    x_train = x_train[..., tf.newaxis].astype(np.float32)

    # plot 10 random images
    example_images = x_train[:10]*127.5 + 127.5
    plot_images(example_images)

    plt.savefig("real_images.png")


    # build the discriminator and the generator
    discriminator = build_discriminator()
    generator = build_generator()


    # build the DCGAN
    dcgan = DCGAN(discriminator=discriminator, generator=generator)

    # compile the DCGAN
    dcgan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  g_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    callbacks = [GenerateSamplesCallback(generator, tf.random.uniform([10, 100]))]
    # train the DCGAN
    dcgan.fit(x_train, epochs=50, batch_size=64, callbacks=callbacks, shuffle=True)

    # generate images
    noise = tf.random.uniform([16, 100])
    generated_images = generator(noise, training=False)
    plot_images(generated_images*127.5 + 127.5, 4, 4)
    plt.savefig("generated_images.png")

    generator.save('generator.h5')

if __name__ == "__main__":
    train_dcgan_mnist()
