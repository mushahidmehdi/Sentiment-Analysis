import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import pickle

def filter_train(sentence):
    split_sent = tf.strings.split(sentence, ',', maxsplit=4)
    test_train = split_sent[1]
    sentiments = split_sent[2]
    return (True if test_train == 'train' and sentiments != 'unsup'
            else False)

def filter_test(sentence):
    split_sent = tf.strings.split(sentence, ',', maxsplit=4)
    test_train = split_sent[1]
    sentiments = split_sent[2]
    return (True if test_train == 'test'and sentiments != 'unsup'
            else False)

ds_train = tf.data.TextLineDataset('imdb.csv').filter(filter_train)
ds_test = tf.data.TextLineDataset('imdb.csv').filter(filter_test)

# Build a Vocab List

tokenizer = tfds.deprecated.text.Tokenizer()

def build_vocab(sentence, threshold=100):
    frequency = {}
    vocabulary = set()
    vocabulary.update('sostoken')
    vocabulary.update('eostoken')

    for world_list in sentence:
        split_sent = tf.strings.split(world_list, ',', maxsplit=4)
        reviews = split_sent[4]
        tokenize_review = tokenizer.tokenize(reviews.numpy().lower())

        for word in frequency:
            if word not in frequency[word]:
                frequency = 1
            else:
                frequency += 1

            if frequency[word] == threshold:
                vocabulary.update(tokenize_review)

    return vocabulary
vocabulary = build_vocab(ds_train)
vocab_list = open('vocabulary.obj', 'bw')
pickle.dump(vocabulary, vocab_list)

# Now Encode the Vocablist

encoder = tfds.deprecated.text.TokenTextEncoder(
    list(vocabulary), oov_token="<UNK>", lowercase=True, tokenizer=tokenizer
)

def my_encoder(text_tensor, label):
    text_encoded = encoder.encode(text_tensor.numpy())
    return text_encoded, label

def map_encoder_func(sentence):
    split_sent = tf.strings.split(sentence, ',', maxsplit=4)
    reviews = "sostoken " + split_sent[4] + " eostoken"
    pos_neg = split_sent[2]
    label = 1 if pos_neg == 'pos' else 0

    (text_encoded, label) = tf.py_function(
        my_encoder, inp=[reviews, label], Tout=(tf.float64, tf.float32)
    )

    text_encoded.shape([None])
    label.shape([])

model = keras.Sequential([
    layers.Masking(),
    layers.Embedding(),
    layers.GlobalAvgPool1D(),
    layers.Dense(10)
])
