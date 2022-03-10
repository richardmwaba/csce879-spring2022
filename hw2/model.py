import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, GRU, Dense, Dropout, Attention, Concatenate


def lstm_rnn(train_ds, **kwargs):
    
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


def lstm_attention(train_ds, **kwargs):
    attended_values = attention(train_ds)

    model = Sequential([
        attention(train_ds),
        tf.keras.layers.LSTMCell(256),
        tf.keras.layers.LSTMCell(64),
        Dense(1)
    ])

    return model


def attention(train_ds, **kwargs):
    
    MAX_TOKENS=5000
    MAX_SEQ_LEN=128
    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=MAX_TOKENS,
        output_mode='int',
        output_sequence_length=MAX_SEQ_LEN)
    VOCAB_SIZE = len(vectorize_layer.get_vocabulary())
    EMBEDDING_SIZE = int(np.sqrt(VOCAB_SIZE))
    embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
    query_layer = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
    value_layer = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
    attention = tf.keras.layers.Attention()
    concat = tf.keras.layers.Concatenate()

    # Vectorize
    encoder = Encoder(train_ds)
#     embeddings = Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True)
    embeddings = embedding_layer(vectorize_layer(train_ds))
    query = query_layer(embeddings)
    value = value_layer(embeddings)
    query_value_attention = attention([query, value])
    attended_values = concat([query, query_value_attention])

    return attended_values
    

def gru_rnn(train_ds, **kwargs):
    
    encoder = Encoder(train_ds)
    model = Sequential([
        encoder,
        Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        Bidirectional(tf.keras.layers.GRU(64,  return_sequences=True)),
        Bidirectional(tf.keras.layers.GRU(32)), # stack 2nd GRU layer
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    return model


def Encoder(train_ds, VOCAB_SIZE=5000):
    encoder = TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_ds.map(lambda text, label: text))  # map function projects out just the first component 'text'
    
    return encoder