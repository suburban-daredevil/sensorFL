import os
import pandas as pd
import keras
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras import layers
from sklearn.model_selection import train_test_split

from helper_functions.print_head_df import print_head_df
from helper_functions.missing_values import missing_values

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model(learning_rate: float = 0.001):
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


fds = None  # Cache FederatedDataset


def load_data():

    '''
    :return: returns the train and test splitted dataset
    '''

    df = pd.read_csv('rpi_20_plus_with_anomalies.csv')
    print_head_df(df)

    '''
    dropping the first column - unwanted
    '''
    df = df.iloc[:, 1:]

    '''
    Checking for NULL Values
    '''

    missing_values(df)

    '''
    Dropping the missing values
    '''
    df.dropna(inplace=True)

    missing_values(df)

    '''
    Train and test split
    '''
    X = df.drop(columns='anomaly', axis=1)
    y = df['anomaly']

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        shuffle=True,
                                                        random_state=6704)

    return X_train, y_train, X_test, y_test