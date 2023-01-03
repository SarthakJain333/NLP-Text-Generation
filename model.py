import numpy as np
import os
import io
import pandas as pd
import tensorflow as tf
from tensorflow import keras

path = 'Twitter-news.txt'
text = open(path, 'rb').read().decode(encoding='utf-8')
print(f'Length of text : {len(text)} characters')

# print(text[:250])

# reading word by word ->
# with open(path, 'r') as file:
#     for line in file:
#         for word in line.split()[:100]:
#             print(word)

vocab = sorted(set(text))
print(f'{len(vocab)} : Length of Vocab')

char2idx = {unique: idx for idx, unique in enumerate(vocab)}
# print(char2idx)

idx2char = np.array(vocab)

text_as_int = np.array([char2idx[char] for char in text])
# print(text_as_int.shape)

sequence_length = 10
examples_per_epoch = len(text) // (sequence_length + 1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# for i in char_dataset.take(5):
#     print(idx2char[i])

sequences = char_dataset.batch(sequence_length+1, drop_remainder=True)

# for i in sequences.take(5):
#     print(''.join(idx2char[i]))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])

    return model

model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

model.compile(
optimizer= keras.optimizers.Adam(), 
loss= [keras.losses.SparseCategoricalCrossentropy(from_logits= True)],
)

checkpoint_dir = 'training_checkpoint_3jan_'
checkpoint_prefix = os.path.join(checkpoint_dir, 'chkpt_{epoch}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

epochs = 5
history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])

model.save('news_model_3jan_.h5')

load_model = tf.keras.models.load_model('news_model_3jan_.h5')

new_model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
new_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
new_model.build(tf.TensorShape([1, None]))

print(new_model.summary())

def generate_txt(model, start_string):
    num_generate = 20
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions/temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    
    return (str(start_string) + ''.join(text_generated))

start_string = input('ENTER THE START STRING: ')
print(generate_txt(new_model, start_string))