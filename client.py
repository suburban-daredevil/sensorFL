import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import flwr as fl
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from helper_functions.print_head_df import print_head_df
from helper_functions.missing_values import missing_values

df = pd.read_csv('rpi_20_plus_with_anomalies.csv')
print_head_df(df)

'''
dropping the first column - unwanted
'''
df = df.iloc[:,1:]

'''
Checking for NULL Values
'''

missing_values(df)

'''
Dropping the missing values
'''
df.dropna(inplace=True)

missing_values(df)

#%%
'''
Train and test split
'''
X = df.drop(columns = 'anomaly', axis = 1)
y = df['anomaly']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    stratify = y,
                                                    shuffle = True,
                                                    random_state = 6704)

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



