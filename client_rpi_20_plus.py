import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import flwr as fl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

X_train = pd.read_csv('rpi_20_plus/x_train.csv')
X_test = pd.read_csv('rpi_20_plus/x_test.csv')
y_train = pd.read_csv('rpi_20_plus/y_train.csv')
y_test = pd.read_csv('rpi_20_plus/y_test.csv')

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.2, verbose=1)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print('Model accuracy on the test dataset:', test_accuracy)

#%%

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=2)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, len(X_test), {'accuracy': accuracy}

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())

#%%



