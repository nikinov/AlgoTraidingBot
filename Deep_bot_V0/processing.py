import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, classification_report
import matplotlib.pylab as plt
import datetime as dt
import time
import pandas as pd
import yahoo_finance as yf
import time


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import Adam

"""
    THIS PART WILL PROCESS THE CLOSING
    PRICE FOR EACH DAY INTO TRAINING
    AND TESTING CHUNKS
"""

def load_stock_close():
    f = open('data/^GSPC.csv', 'r').readlines()[1:]
    raw_data = []
    for line in f:
        close_price = float(line.split(',')[4])
        raw_data.append(close_price)

    return raw_data

def split_into_chunks(data, train_size=20 ,scale=False):
    X, Y = [], []
    if scale:
        data = preprocessing.scale(data)
    for i in range(len(data)):
        if i+train_size+1 < len(data):
            x_i = data[i:i+train_size]
            y_i = data[i+train_size+1]

            X.append(x_i)
            Y.append(y_i)

    return np.array(X), np.array(Y)

def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def create_test_data(data, percentage=0.8):
    X, y = data
    X_train = X[0:int(len(X) * percentage)]
    Y_train = y[0:int(len(y) * percentage)]
    X_train, Y_train = shuffle_in_unison(X_train, Y_train)

    X_test = X[int(len(X) * percentage):]
    Y_test = y[int(len(X) * percentage):]

    return X_train, X_test, Y_train, Y_test

# defining parameters
TRAIN_SIZE = 20

# make a training and testing set with 20 days of closing price for training and the 21st closing price as validation data
X_train, X_test, Y_train, Y_test = create_test_data(
    split_into_chunks(load_stock_close(),train_size=TRAIN_SIZE), percentage=0.9
)

"""
    IN THIS NEXT PART WE WILL BE WORKING WITH OUR MODEL
"""

# Define the model.
model = Sequential([
    Dense(500, input_shape=(TRAIN_SIZE,), activation='relu'),
    Dropout(.25),
    Dense(250, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model.
model.compile(Adam(lr=.0001), loss="mse", metrics=["accuracy"])

# Now let's train the model
model.fit(X_train, Y_train, batch_size=10197, epochs=15, verbose=1, validation_split=.1)

# Evaluate our model
prediction = model.predict(X_test[0:1],verbose=0)
print(prediction)
print(Y_test[0])









