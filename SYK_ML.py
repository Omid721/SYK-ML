import time
import warnings
import pickle
import io
import requests
import zipfile
import contextlib  # For redirecting stdout
import sys

# Importing necessary libraries for numerical computations and machine learning
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
import scipy.optimize  # For optimization
import scipy.stats  # For statistical distributions
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.manifold import (MDS, LocallyLinearEmbedding, Isomap,
                               SpectralEmbedding, TSNE)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import (RidgeClassifier, SGDClassifier,
                                   Ridge, Lasso, LinearRegression,
                                   LogisticRegression)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                              RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)

# Define the Pauli matrices
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

# Define the Sachdev-Ye-Kitaev (SYK) Hamiltonian
def syk_hamiltonian(N, disorder_strength):
    """Construct the SYK Hamiltonian for a given number of Majorana fermions.

    Args:
        N (int): Number of Majorana fermions (must be even).
        disorder_strength (float): Strength of disorder in the system.

    Returns:
        np.ndarray: The SYK Hamiltonian matrix.
    """
    if N % 2 != 0:
        raise ValueError("Number of Majorana fermions (N) should be even.")

    # Generate random couplings
    J = disorder_strength * np.random.randn(N, N, N, N) / np.sqrt(2)

    # Initialize the Hamiltonian matrix
    H = np.zeros((2**N, 2**N), dtype=np.complex128)

    # Construct the Hamiltonian
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                for l in range(k + 1, N):
                    H += J[i, j, k, l] * 1j * np.kron(
                        np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(
                            np.kron(np.eye(2**(i)), sigma_y), np.eye(2**(j - i - 1))),
                            sigma_y), np.eye(2**(k - j - 1))),
                            sigma_y), np.eye(2**(l - k - 1))),
                            sigma_y), np.eye(2**(N - l - 1)))

    return H

# Generate training data
N = 6  # Number of Majorana fermions (even for SYK)
disorder_strength = 1
num_samples = 100

def haar_random_state(dimension):
    """Generate a Haar random state.

    Args:
        dimension (int): Dimension of the Hilbert space.

    Returns:
        np.ndarray: Normalized Haar random state vector.
    """
    random_vector = np.random.normal(0, 1, size=(dimension,))
    normalized_vector = random_vector / np.linalg.norm(random_vector)

    return normalized_vector

# Set the dimension of the Hilbert space
hilbert_space_dimension = 2 ** N

def my_gen(N_SYK_sample, N_Haar_sample):
    """Generate training data consisting of Haar random states and SYK states.

    Args:
        N_SYK_sample (int): Number of SYK samples to generate.
        N_Haar_sample (int): Number of Haar random samples to generate.

    Returns:
        pd.DataFrame: DataFrame containing generated states and labels.
    """
    X = np.empty((N_Haar_sample, hilbert_space_dimension + 1))
    y = np.zeros((N_SYK_sample, hilbert_space_dimension + 1))
    z = np.zeros((N_SYK_sample + N_Haar_sample, hilbert_space_dimension + 1))

    for k in range(max(N_SYK_sample, N_Haar_sample)):
        if k < N_Haar_sample:
            X[k, 0:hilbert_space_dimension] = haar_random_state(hilbert_space_dimension)
            X[k, hilbert_space_dimension] = 0
            z[k + N_SYK_sample, :] = X[k, :]
        if k < N_SYK_sample:
            H = syk_hamiltonian(N, disorder_strength)
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            y[k, 0:hilbert_space_dimension] = eigenvectors[0] / np.linalg.norm(eigenvectors[0])
            y[k, hilbert_space_dimension] = 1
            z[k, :] = y[k, :]

    column_names = [f'C_WF_{i}' for i in range(hilbert_space_dimension)]
    z1 = z[:, 0:hilbert_space_dimension]
    z2 = z[:, hilbert_space_dimension]
    z1 = pd.DataFrame(z1, columns=column_names)
    z2 = pd.DataFrame(z2 > 0.1, columns=['is SYK'])

    return pd.concat([z1, z2], axis=1)

# Generate sample data
data = my_gen(500, 500)
data = sk.utils.shuffle(data, random_state=0)
target = 'is SYK'
features = data.columns.drop([target])
y_train = data[target].astype(int)
x_train = data[features]

# Define the neural network model
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
        tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    ])
    return model

# Build and compile the neural network model
model = build_nn_model(len(features))
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Display model summary
model.summary()

# Function for generating test data
def test_gen(a, N):
    """Generate test data based on the SYK Hamiltonian.

    Args:
        a (float): Parameter for Hamiltonian evolution.
        N (int): Number of Majorana fermions.

    Returns:
        pd.DataFrame: DataFrame containing generated test states.
    """
    X = np.empty((1, hilbert_space_dimension), dtype=complex)
    psi = haar_random_state(hilbert_space_dimension)
    H = syk_hamiltonian(N, disorder_strength)

    for i in range(N):
        X[0, :] += (1 / np.math.factorial(i)) * (a * H) ** i @ psi

    X_real = X.real.astype(float)
    column_names = [f'C_WF_{i}' for i in range(hilbert_space_dimension)]
    X = pd.DataFrame(X_real, columns=column_names)

    return X

# Generate test data
data_test = my_gen(50, 50)
y_test = data_test[target].astype(int)
x_test = data_test[features]

x_test_our_case = test_gen(2, N)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predict the ground state using the trained neural network
decision_values_NN = model.predict(x_test)
our_case_prediction = model.predict(x_test_our_case)

print("Predicted probabilities for being SYK:", decision_values_NN)
print("=" * 80)
print("Predicted probabilities for being SYK in our case:", our_case_prediction)

# Handling Google Drive for data retrieval
from google.colab import drive
drive.mount('/content/drive')

# Path to the uploaded zip file
zip_file_path = '/content/drive/MyDrive/Data/data_SYK_N_6.zip'
file_to_read = 'data'

# Read the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as archive:
    with archive.open(file_to_read) as file:
        data_N6 = pd.read_csv(file)

# Now 'data_N6' contains the DataFrame with the content of the specified CSV file
print(data_N6)
print(data_N6.describe())

# Prepare datasets for different N values
test_size = 0.05

# Shuffle and split data for N = 6
data_N6 = sk.utils.shuffle(data_N6, random_state=0)
target = 'is SYK'
features_N6 = data_N6.columns.drop([target])
y_N6 = data_N6[target].astype(int)
x_N6 = data_N6[features_N6]

x_train_N6, x_test_N6, y_train_N6, y_test_N6 = sk.model_selection.train_test_split(
    x_N6, y_N6, test_size=test_size, random_state=0)

# Train and evaluate on N = 6 data
model_N6 = build_nn_model(len(features_N6))
model_N6.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
model_N6.fit(x_train_N6, y_train_N6, epochs=50, batch_size=32)
loss_N6, accuracy_N6 = model_N6.evaluate(x_test_N6, y_test_N6)

print(f"Test Loss for N=6: {loss_N6}")
print(f"Test Accuracy for N=6: {accuracy_N6}")

# Generate predictions for the N=6 test set
decision_values_NN_N6 = model_N6.predict(x_test_N6)
print("Predicted probabilities for being SYK in N=6:", decision_values_NN_N6)
