"""
Anubhav Natani
2019
Production Model
Tensorflow 2.0
"""
import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, input_size, batch_size, hidden_size, embedding_dim, gru_units):
        # some what different as compared to torch model
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        # vocab size
        self.input_size = input_size
        # embedding dimension
        self.embedding = tf.keras.layers.Embedding(input_size, embedding_dim)
        # units for gru
        self.gru_units = gru_units
        # defing gru layer
        self.gru = tf.keras.layers.GRU(
            self.gru_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")

    # on calling
    def call(self, x, hidden):
        # here x refer as the input data same as the torch
        x = self.embedding(x)
        # gru returns state and output
        out, state = self.gru(x, initial_state=hidden)
        # first state is initialized below and then the state changes according to the state given by gru
        return out, state
    # function for initial state initialization

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.gru_units))

# Attention Layer in Between
