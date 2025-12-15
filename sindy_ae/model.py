
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class SindyAutoencoder(Model):
    def __init__(self, latent_dim=3, sindy_library_dim=10, sindy_reg=1e-3):
        super().__init__()
        self.latent_dim = latent_dim
        self.sindy_library_dim = sindy_library_dim
        self.sindy_reg = sindy_reg

        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(None,)),
            layers.Dense(64, activation='tanh'),
            layers.Dense(32, activation='tanh'),
            layers.Dense(latent_dim)
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(32, activation='tanh'),
            layers.Dense(64, activation='tanh'),
            layers.Dense(1)
        ])

        # SINDy coefficient matrix
        self.coeffs = tf.Variable(tf.random.normal((latent_dim, sindy_library_dim)), trainable=True)

    def sindy_library(self, z):
        poly_terms = [z, z**2, tf.sin(z), tf.cos(z)]
        concat = tf.concat(poly_terms, axis=-1)
        return concat[:, :self.sindy_library_dim]

    def call(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)

        # Compute SINDy prediction
        lib = self.sindy_library(z)
        dz_dt_pred = tf.matmul(lib, tf.transpose(self.coeffs))
        return x_rec, z, dz_dt_pred

    def train_step(self, data):
        x, dx_dt = data

        with tf.GradientTape() as tape:
            x_rec, z, dz_dt_pred = self(x)

            rec_loss = tf.reduce_mean((x - x_rec)**2)
            sindy_loss = tf.reduce_mean((dx_dt - dz_dt_pred)**2)
            reg_loss = self.sindy_reg * tf.reduce_sum(tf.abs(self.coeffs))

            loss = rec_loss + sindy_loss + reg_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": loss, "rec_loss": rec_loss, "sindy_loss": sindy_loss}

