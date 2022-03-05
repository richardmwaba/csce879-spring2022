import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense, Dropout, Attention, Concatenate


def simple_rnn(train_ds, **kwargs):
    
    encoder = Encoder(train_ds)
    model = Sequential([
        encoder,
        Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),  # return_sequences needs to be True for stacking subsequent LSTM layers
        Bidirectional(tf.keras.layers.LSTM(32)), # stack 2nd LSTM layer
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    return model


def Encoder(train_ds, VOCAB_SIZE=5000):
    encoder = TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_ds.map(lambda text, label: text))
    
    return encoder