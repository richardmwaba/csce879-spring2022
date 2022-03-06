from sklearn import preprocessing
import tensorflow as tf
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Attention, Concatenate
# from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Attention, Concatenate


def simple_rnn(train_text, **kwargs):
    
    encoder = Encoder(train_text)
    model = Sequential([
        encoder,
        Embedding(len(encoder.get_vocabulary()), 64, mask_zero=False),
        Bidirectional(LSTM(64,  return_sequences=True)),  # return_sequences needs to be True for stacking subsequent LSTM layers
        Bidirectional(LSTM(32)), # stack 2nd LSTM layer
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    return model


def Encoder(train_text, VOCAB_SIZE=5000):
    encoder = TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_text)
    
    return encoder