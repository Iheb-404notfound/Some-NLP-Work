import string

import tensorflow as tf
import matplotlib as plt
import numpy as np
import pandas as pd

# Explore the data
print('#################### Exploring The Data #######################')

data = pd.read_csv('spotify_millsongdata.csv')[:10]

print(data)
data.describe()
data.info()


def clean(dataset):
    # Remove punctuations and unwanted expressions
    dataset.text = dataset.text.str.replace(f'[{string.punctuation}\r]', '')

    # Make all the data lower case
    dataset.text = dataset.text.str.lower()

    # Concatenate all lines and make a lyrics
    lyrics = dataset.text.str.cat()

    # Split the lines
    corpus = lyrics.split('\n')

    # Remove the trailing spaces
    corpus = [l.rstrip() for l in corpus]

    # Remove the empty lines
    corpus = [l for l in corpus if l != '']

    return corpus


print(data)
data = clean(data)

# Tokenize the data
toker = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
toker.fit_on_texts(data)

tokenized_data = toker.texts_to_sequences(data)
# print(tokenized_data)
# Create the 1-grams
features = []
labels = []
max_len = 700
vocab_size = len(toker.word_index) + 1
for seq in tokenized_data:
    max_len = max(max_len, len(seq))
    for i in range(1, len(seq)):
        features.append(seq[:i])
        labels.append(seq[i])

features = tf.constant(tf.keras.preprocessing.sequence.pad_sequences(features, max_len, padding='post'))
labels = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)
print(features, labels)

# Preparing the train and test sets
split = int(0.85 * len(features))

train_features, train_labels = features[:split], labels[:split]
test_features, test_labels = features[split:], labels[split:]

# Creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

# Show the model architecture
model.summary()

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Fit the model
model.fit(
    features,
    labels,
    epochs=200,
)

# reload the best weights
model.load_weights('checks/check.ckpt')

# make some predictions
text_seed = 'Hey'
print(text_seed, end=' ')
for i in range(10):
    inp = tf.constant(
        tf.keras.preprocessing.sequence.pad_sequences(toker.texts_to_sequences([text_seed]), max_len, padding='post'))
    out = tf.argmax(model.predict(inp, verbose=0), axis=-1)
    for word, index in toker.word_index.items():
        if index == out:
            output_word = word
            print(word, '-')
            break
    print(output_word, end=' ')
    text_seed += ' ' + output_word
