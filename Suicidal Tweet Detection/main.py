import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import matplotlib.pyplot as plt

data = pd.read_csv('suicide_ds.csv')
print(data)
data.describe()
data.Suicide.info()

# Preprocess the data
# make it a binary classification data
data.Suicide = data.Suicide.apply(lambda x: float(x == 'Potential Suicide post '))
print(data)

# Get features and labels
features = tf.constant(data.Tweet.to_numpy(dtype=str))
labels = tf.constant(data.Suicide)

# divide them into train and test/validation sets
split = int(0.85 * len(features))
train_features, train_labels = features[:split], labels[:split]
test_features, test_labels = features[split:], labels[split:]

# prefetch the datasets
train_data = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(32).prefetch(tf.data.AUTOTUNE)
test_data = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(32).prefetch(tf.data.AUTOTUNE)

# Create the text vectorization
vocabs = 10000
max_sequence = 50
vectorization = tf.keras.layers.TextVectorization(vocabs,
                                                  'lower_and_strip_punctuation',
                                                  'whitespace',
                                                  None,
                                                  'int',
                                                  max_sequence)

vectorization.adapt(features)

print(features[3], vectorization([features[3]]))

#url = 'https://tfhub.dev/google/Wiki-words-250/2'
#hub_layer = hub.KerasLayer(url, trainable=False, input_shape=(), dtype=tf.string)
#print(features[0], hub_layer([features[0]]))

# Create the model
model = tf.keras.Sequential([
    vectorization,
    tf.keras.layers.Embedding(vocabs, 128),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True), name='bi1'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False), name='b2'),
    tf.keras.layers.Dense(1, 'sigmoid')
])

# model = tf.keras.Sequential([
#     hub_layer,
#     tf.keras.layers.Reshape((1, 250)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True), name='bi1'),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False), name='b2'),
#     tf.keras.layers.Dense(1, 'sigmoid')
# ])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(
    train_data,
    epochs=5,
    steps_per_epoch=len(train_data),
    validation_data=test_data,
    validation_steps=len(test_data)
)
