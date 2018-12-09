import pdb
import sys
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pickle

from technical import load_full_data_set


########################################################################

# Load Data

prices, volumes, labels, tickers, dates = load_full_data_set()
assert(len(prices) == len(volumes) == len(labels) == len(dates) == len(tickers))

train_data = prices  #@TODO toggle this between prices and volumes

########################################################################

# Normalize Data

NUM_INPUT_VALS = 10000

for indx in range(len(train_data)):
    # move the mean to zero
    # change each value to be num standard deviations from zero
    mean = np.mean(train_data[indx])
    std = np.std(train_data[indx])
    train_data[indx] = (train_data[indx] - mean) / std
    # make the largest and smallest values fit within 0 to 1000 int range
    max_val = max(max(train_data[indx]), abs(min(train_data[indx])))
    scale_factor = (NUM_INPUT_VALS - 1) / max_val / 2
    train_data[indx] = train_data[indx] * scale_factor
    # set new center at 500 instead of zero
    train_data[indx] += NUM_INPUT_VALS / 2
    train_data[indx] = np.clip(train_data[indx], 0, NUM_INPUT_VALS-1)

train_data = train_data.astype(int) 

# validate range of all training data
for indx in range(len(train_data)):
    max_val = max(train_data[indx])
    min_val = min(train_data[indx])
    if max_val >= NUM_INPUT_VALS or min_val < 0:
        print("Train data index %s value %s or %s is out of bounds!" %\
              (indx, max_val, min_val))

print("Normalize any input data to the model thus:")
print("train_data[indx] = (train_data[indx] - mean_for_indx) / std_for_indx.")
print("Scale train_data to range 0 to 999 and convert to integers.")


########################################################################

# Split Off Train and Test Data

indx_80_percent = int(len(train_data) * 0.8)
test_data = train_data[indx_80_percent:]
test_labels = labels[indx_80_percent:]
test_dates = dates[indx_80_percent:]
test_tickers = tickers[indx_80_percent:]
assert(len(test_data) == len(test_labels) == len(test_dates) == len(test_tickers))

train_data = train_data[:indx_80_percent]
train_labels = labels[:indx_80_percent]
train_dates = dates[:indx_80_percent]
train_tickers = tickers[:indx_80_percent]
assert(len(train_data) == len(train_labels) == len(train_dates) == len(train_tickers))


########################################################################

# Build Model

NUM_NODES = 4 
model = keras.Sequential()
model.add(keras.layers.Embedding(NUM_INPUT_VALS, NUM_NODES))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(NUM_NODES, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# Embedding layer turns positive integers into dense vectors of size 4
# e.g. [[1], [2], ...] becomes [[a,b,c,...,z], [a,b,c,...,z], ...]
# GlobalAveragePooling1D averages the size-4 input vectors from the Embedding layer
# Dense 4 = fully-connected layer with 4-hidden units
# Dense 1 = single-output, fully connected layer.

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Built Model")


########################################################################

# Validation Set

indx_30_percent = int(len(train_data) * 0.30)

data_validate = train_data[:indx_30_percent]
lbl_validate = train_labels[:indx_30_percent]
part_train_data = train_data[indx_30_percent:]
part_train_lbl = train_labels[indx_30_percent:]


########################################################################

# Train Model

history = model.fit(part_train_data,
                    part_train_lbl,
                    epochs=40,
                    batch_size=512,
                    validation_data=(data_validate, lbl_validate),
                    verbose=1)

########################################################################


# Test Model

results = model.evaluate(test_data, test_labels)
print('Tested model.  Loss & accuracy = %s.' % results)
# binary_crossentropy loss = https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
# accuracy = % labels matched correctly


########################################################################


# Plot Accuracy and Loss Over Time
history_dict = history.history

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # clear figure
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

