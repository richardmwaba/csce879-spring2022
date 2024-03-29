from sklearn import preprocessing
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, GRU, Dense, Dropout, Attention, Concatenate


def lstm_attention(train_ds, **kwargs):
    dense_units = kwargs['dense_units'] if 'dense_units' in kwargs else 64
    
    encoder = Encoder(train_ds)
    
    VOCAB_SIZE = len(encoder.get_vocabulary())
    EMBEDDING_SIZE = int(np.sqrt(VOCAB_SIZE))
    embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
    
    query_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        padding='same')
    value_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        padding='same')
    
    attention = tf.keras.layers.Attention()
    concat = tf.keras.layers.Concatenate()
    
    cells = [tf.keras.layers.LSTMCell(32), tf.keras.layers.LSTMCell(8)]
    rnn = tf.keras.layers.RNN(cells)
    dense = Dense(dense_units, activation='relu')
    dropout = Dropout(0.1)
    output_layer = tf.keras.layers.Dense(1)
    
    input_x = tf.keras.Input(shape=(1,), dtype=tf.string)  # input is string of one review
    embeddings = embedding_layer(encoder(input_x))
    query = query_layer(embeddings)
    value = value_layer(embeddings)
    query_value_attention = attention([query, value])
    attended_values = concat([query, query_value_attention])
    logits = output_layer(dropout(dense(rnn(attended_values))))
    model = tf.keras.Model(input_x, logits)
    
    return model


def gru_attention(train_ds, **kwargs):
    dense_units = kwargs['dense_units'] if 'dense_units' in kwargs else 64
    
    encoder = Encoder(train_ds)
    
    VOCAB_SIZE = len(encoder.get_vocabulary())
    EMBEDDING_SIZE = int(np.sqrt(VOCAB_SIZE))
    embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
    
    query_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        padding='same')
    value_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        padding='same')
    
    attention = tf.keras.layers.Attention()
    concat = tf.keras.layers.Concatenate()
    
    cells = [tf.keras.layers.GRUCell(32), tf.keras.layers.GRUCell(8)]
    rnn = tf.keras.layers.RNN(cells)
    dense = Dense(dense_units, activation='relu')
    dropout = Dropout(0.1)
    output_layer = tf.keras.layers.Dense(1)
    
    input_x = tf.keras.Input(shape=(1,), dtype=tf.string)  # input is string of one review
    embeddings = embedding_layer(encoder(input_x))
    query = query_layer(embeddings)
    value = value_layer(embeddings)
    query_value_attention = attention([query, value])
    attended_values = concat([query, query_value_attention])
    logits = output_layer(dropout(dense(rnn(attended_values))))
    model = tf.keras.Model(input_x, logits)
    
    return model


def Encoder(train_ds, VOCAB_SIZE=5000):
    encoder = TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_ds.map(lambda text, label: text))  # map function projects out just the first component 'text'
    
    return encoder
