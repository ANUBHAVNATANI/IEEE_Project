"""
Anubhav Natani
2019
Production Model
Tensorflow 2.0
"""
import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, input_size, batch_size, hidden_size, embedding_dim, enc_units):
        # some what different as compared to torch model
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        # vocab size
        self.input_size = input_size
        # embedding dimension
        self.embedding = tf.keras.layers.Embedding(input_size, embedding_dim)
        # units for gru
        self.enc_units = enc_units
        # defing gru layer
        self.gru = tf.keras.layers.GRU(
            self.enc_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")

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
        return tf.zeros((self.batch_size, self.enc_units))

# Attention Layer in Between
# Calculating attention weights and then context vector and finally attention vector


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        # small nueral network to predict the intermediate vector e for the attention mechanism
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden step we use to decide the the input to the small nueral network
        hidden_state_att = tf.expand_dims(query, 1)
        # this is done for the manipulation of the input to the nureal network so that proper dimension can be maintained
        # e = FC(tanh(FC(Encodero/p)+FC(Decoder Hidden State)))
        # basic nureal network arch
        # values is the encoder o/p
        e = self.V(tf.nn.tanh(self.W1(values)+self.W2(hidden_state_att)))
        # attention weights
        # softmax to the score
        att_weights = tf.nn.softmax(e, axis=1)
        # context vector i.e weighted sum
        cont_vec = att_weights*values
        # multiplication with the matrix vectorization
        cont_vec = tf.reduce_sum(cont_vec, axis=1)
        # after mutliplication we need to sum all of them using the first axis
        return cont_vec, att_weights


class Decoder(tf.keras.Model):
    def __init__(self, input_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(input_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        self.fc = tf.keras.layers.Dense(input_size)
        # initializing the attention layer
        self.attn = Attention(self.dec_units)

    def call(self, x, hidden, enc_op):
        # run on the encoder output
        cont_vec, att_wgt = self.attn(hidden, enc_op)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(cont_vec, 1), x], axis=-1)
        # both attention and embedded input goes in the gru input
        out, state = self.gru(x)
        # output shape change
        out = tf.reshape(out, (-1, out.shape[2]))
        # passing this output to the fully connected layer
        x = self.fc(out)

        # state pass for the next state and attention weights for the next state
        return x, state, att_wgt
