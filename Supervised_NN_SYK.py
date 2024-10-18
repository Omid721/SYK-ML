import time
import warnings
import pickle
import io
import requests
import zipfile
import contextlib  # For redirecting stdout
import sys

# Importing libraries for numerical computations and machine learning
import multiprocessing as mp
import numpy as np
import scipy as sp
import pandas as pd
import sympy as sym
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
import sklearn as sk

# Importing submodules from sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import (RidgeClassifier, SGDClassifier, Ridge, Lasso,
                                   LinearRegression, LogisticRegression)
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import (KNeighborsClassifier, KNeighborsRegressor)
from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                              RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.manifold import (MDS, LocallyLinearEmbedding, Isomap,
                               SpectralEmbedding, TSNE)
from sklearn.model_selection import train_test_split
import sklearn.utils

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data from Google Drive
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define file paths for the datasets
zip_file_paths = {
    'N6': '/content/drive/MyDrive/Data/data_SYK_N6.zip',
    'N8': '/content/drive/MyDrive/Data/data_SYK_N8.zip',
    'N10': '/content/drive/MyDrive/Data/data_SYK_N10.zip',
    'N12': '/content/drive/MyDrive/Data/data_SYK_N12.zip'
}

# Function to read data from ZIP files
def load_data(zip_path):
    """Load data from a specified ZIP file.

    Args:
        zip_path (str): Path to the ZIP file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    with zipfile.ZipFile(zip_path, 'r') as archive:
        with archive.open('data') as file:
            return pd.read_csv(file)

# Load datasets
data = {key: load_data(path) for key, path in zip_file_paths.items()}

# Prepare data for training and testing
test_size = 0.05
datasets = {}

for key in data.keys():
    shuffled_data = sk.utils.shuffle(data[key], random_state=0)
    target = 'is SYK'
    features = shuffled_data.columns.drop([target, 'Unnamed: 0'])
    y = shuffled_data[target].astype(int)
    X = shuffled_data[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=test_size)

    datasets[key] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'features': features
    }

# Function to build a neural network model
def build_nn_model(input_shape):
    """Build a neural network model for binary classification.

    Args:
        input_shape (int): Number of input features.

    Returns:
        tf.keras.Sequential: Compiled neural network model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    return model

# Create and compile models for each dataset
models = {}
for key, dataset in datasets.items():
    model = build_nn_model(len(dataset['features']))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    model.fit(dataset['X_train'], dataset['y_train'], epochs=50, batch_size=32)
    models[key] = model

# Evaluate models and gather loss and accuracy metrics
losses = {}
accuracies = {}
for key, model in models.items():
    loss, accuracy = model.evaluate(datasets[key]['X_test'], datasets[key]['y_test'])
    losses[key] = loss
    accuracies[key] = accuracy

# Print loss and accuracy
print("Losses:", losses)
print("Accuracies:", accuracies)
print("=" * 80)

# Visualize loss and accuracy
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot Loss
ax1.bar(losses.keys(), losses.values(), color='red')
ax1.set_title('Loss')
ax1.set_ylabel('Loss Value')

# Plot Accuracy
ax2.bar(accuracies.keys(), accuracies.values(), color='blue')
ax2.set_title('Accuracy')
ax2.set_ylabel('Accuracy Value')

plt.tight_layout()
plt.show()
