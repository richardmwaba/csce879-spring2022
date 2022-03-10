from sklearn import preprocessing
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, GRU, Dense, Dropout, Attention, Concatenate


def lstm_rnn(train_ds, **kwargs):
    dense_units = kwargs['dense_units'] if 'dense_units' in kwargs else 64
    
    encoder = Encoder(train_text)
    model = Sequential([
        encoder,
        Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),  # return_sequences needs to be True for stacking subsequent LSTM layers
        Bidirectional(tf.keras.layers.LSTM(32)), # stack 2nd LSTM layer
        Dense(dense_units, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    return model


def gru_rnn(train_ds, **kwargs):
    dense_units = kwargs['dense_units'] if 'dense_units' in kwargs else 64
    
    encoder = Encoder(train_ds)
    model = Sequential([
        encoder,
        Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        Bidirectional(tf.keras.layers.GRU(64,  return_sequences=True)),
        Bidirectional(tf.keras.layers.GRU(32)), # stack 2nd GRU layer
        Dense(dense_units, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    return model


def Encoder(train_text, VOCAB_SIZE=5000):
    encoder = TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_ds.map(lambda text, label: text))  # map function projects out just the first component 'text'
    
    return encoder