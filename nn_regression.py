import pdb
import sys
from math import sqrt

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fundamentals import load_full_data_set as load_fundamentals_data
from fundamentals import METRIC_NAMES

# sudo apt-get install python-tk
# sudo apt-get install python-gtk2-dev
# pip install -U matplotlib


########################################################################

# Get and Clean Data

train_data, train_labels, year_ranges = load_fundamentals_data()
assert(len(train_data) == len(train_labels))
assert(len(year_ranges) == len(train_labels))
print("Pre-clean None, num examples: %s" % len(train_data))

indx = 0
while indx < len(train_data):
    if None in train_data[indx] or not train_labels[indx]:
        # delete rows with None
        train_data = np.delete(train_data, indx, axis=0)
        train_labels = np.delete(train_labels, indx, axis=0)
        year_ranges = np.delete(year_ranges, indx, axis=0)
        continue
    indx += 1

assert(len(train_data) == len(train_labels))
assert(len(year_ranges) == len(train_labels))
print("Post-clean None, num examples: %s" % len(train_data))


########################################################################

# Normalize Data

mean = train_data.mean(axis=0)

# std = train_data.std(axis=0) --> gives error below, so calculating manually
# AttributeError: 'float' object has no attribute 'sqrt'
mean_l = mean.tolist()
train_data_l = train_data.tolist()
std_l = []
num_cols = len(train_data_l[0])
num_rows = len(train_data_l)
for col in range(num_cols):
    sum_diff_squared = 0
    for row in range(num_rows):
        sum_diff_squared += (train_data[row][col] - mean_l[col])**2
    col_std = sqrt(sum_diff_squared / num_rows)
    std_l.append(col_std)
std = np.array(std_l)
assert(len(std) == len(mean))
assert(len(METRIC_NAMES) == len(mean))
assert(len(train_data) == len(train_labels))
assert(len(year_ranges) == len(train_labels))
train_data = (train_data - mean) / std

print("Normalize any input data to the model thus:")
print("metric_val = (metric_val - metric_mean_val) / metric_std")
print("\n\nWhere the metric means and standard deviations are:")
for indx in range(len(METRIC_NAMES)):
    print("    %s, mean is, %s, std is, %s" % (METRIC_NAMES[indx], mean[indx], std[indx]))


#######################################################################

# Shuffle Data

order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]
year_ranges = year_ranges[order]


########################################################################

# Use 20% as Test Data and 80% as Train Data

indx_80_percent = int(len(train_data) * 0.8)

test_data = train_data[indx_80_percent:]
test_labels = train_labels[indx_80_percent:]
test_years = year_ranges[indx_80_percent:]
assert(len(test_data) == len(test_labels) == len(test_years))

train_data = train_data[:indx_80_percent]
train_labels = train_labels[:indx_80_percent]
train_years = year_ranges[:indx_80_percent]
assert(len(train_data) == len(train_labels) == len(train_years))


########################################################################

# Create model

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(1, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
    keras.layers.Dense(52, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)
  model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
  return model

# loss 'mse' = Mean Squared Error = sum(errors^2) / num_errors
# metrics 'mae' = Mean Absolute Error = sum(|errors|) / num_errors

model = build_model()
model.summary()


########################################################################

# Train Model

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0:
        print('')
    sys.stdout.write('.')

EPOCHS = 500

history = model.fit(train_data,
                    train_labels,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[PrintDot()])


########################################################################

# Show Training Results

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error (price change next year)')
    # error on not-yet-trained-upon data
    plt.plot(history.epoch,
           np.array(history.history['mean_absolute_error']),
           label='Training MAE')
    # error on validation data
    plt.plot(history.epoch,
           np.array(history.history['val_mean_absolute_error']),
           label = 'Validation MAE')
    plt.legend()
    plt.ylim([0, 100])
    plt.show()

plot_history(history)


########################################################################

# Testing vs Test Set

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("\nTesting model against test data:")
print("Mean Squared Error: %s %%" % round(loss, 3))
print("Mean Absolute Error: %s %%" % round(mae, 3))


########################################################################

# Predict Price Changes

test_predictions = model.predict(test_data).flatten()

assert(len(test_predictions) == len(test_labels))
num_matching_signs = 0.0
for indx in range(len(test_labels)):
    if test_predictions[indx] < 0 and test_labels[indx] < 0:
        num_matching_signs += 1
    elif test_predictions[indx] > 0 and test_labels[indx] > 0:
        num_matching_signs += 1
percent_sign_matches = num_matching_signs / len(test_labels) * 100
print("Percent correct predictions of price change direction: %s %%" %\
      round(percent_sign_matches, 3))


# Mean Absolute Error
sum_abs_diffs = 0
for indx in range(len(test_labels)):
    sum_abs_diffs += abs(test_predictions[indx] - test_labels[indx])
print("Mean absolute error for test data: %s %%" %\
      round((sum_abs_diffs / len(test_labels)), 3))


plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-1000, 1000], [-1000, 1000])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
plt.show()


