from tensorflow.keras.layers import Input, Dense, LSTM, LeakyReLU, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

def build_generator(latent_dim, sequence_length):
    generator = Sequential()
    generator.add(Dense(128, activation="relu", input_dim=latent_dim))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Dense(sequence_length, activation="linear"))
    return generator

def build_discriminator(sequence_length):
    discriminator = Sequential()
    discriminator.add(Dense(128, activation="relu", input_shape=(sequence_length,)))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(1, activation="sigmoid"))
    discriminator.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5), metrics=["accuracy"])
    return discriminator

def build_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze discriminator during generator training
    gan_input = Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))
    return gan

latent_dim = 100  # Dimensionality of the random noise input
sequence_length = 100  # Assuming 100 time steps per sequence

generator = build_generator(latent_dim, sequence_length)
discriminator = build_discriminator(sequence_length)
gan = build_gan(generator, discriminator)

def train_gan(generator, discriminator, gan, epochs=10000, batch_size=32, sample_interval=1000):
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_seqs = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_seqs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_seqs, real)
        d_loss_fake = discriminator.train_on_batch(generated_seqs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real)

        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")

train_gan(generator, discriminator, gan)

def generate_synthetic_data(generator, num_samples):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_data = generator.predict(noise)
    return generated_data

synthetic_data = generate_synthetic_data(generator, 1000)  
