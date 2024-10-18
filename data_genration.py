# Import necessary libraries
import time
import warnings
import pickle
import io
import requests
import zipfile
import contextlib  # For redirecting stdout
import sys
import multiprocessing as mp
import numpy as np
import scipy as sp
import pandas as pd
import sympy as sym
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
from sklearn import (utils, preprocessing, model_selection, metrics)
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.manifold import (MDS, LocallyLinearEmbedding, Isomap, SpectralEmbedding, TSNE)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import (RidgeClassifier, SGDClassifier, Ridge, Lasso, LinearRegression, LogisticRegression)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)

# Define the Pauli matrices
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

def tensor(a, i, N, p=2):
    """
    Create a tensor product of matrix 'a' at the ith position in a Hilbert space of N qudits.

    Parameters:
        a (ndarray): pxp Unitary matrix for a single qudit.
        i (int): Position in the tensor product (1-indexed).
        N (int): Total number of qudits.
        p (int): Degree of freedom per site (default is 2 for qubits).

    Returns:
        A (ndarray): Resulting matrix after tensor product.
    """
    dim1 = int(p ** (i - 1))
    dim2 = int(p ** (N - i))
    return np.kron(np.kron(np.identity(dim1), a), np.identity(dim2))

def majorana(N):
    """
    Create a representation of N Majorana fermions.

    Parameters:
        N (int): Number of Majorana fermions (must be even).

    Returns:
        Xi (ndarray): Representation of Majorana fermions.
    """
    if N % 2 != 0:
        raise ValueError("N must be even.")

    # SU(2) matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    dim = int(2 ** (N / 2))
    Xi = np.zeros((dim, dim, N), dtype=complex)

    Zs = np.eye(dim, dtype=complex)

    for i in range(1, N + 1):
        xi = tensor(X if i % 2 == 1 else Y, (i + 1) // 2 if i % 2 == 1 else i // 2, N // 2)
        Xi[:, :, i - 1] = np.dot(Zs, xi)

        if i % 2 == 0:
            Zs = np.dot(Zs, tensor(Z, i // 2, N // 2))

    return Xi / np.sqrt(2)  # Kitaev's normalization

def syk_hamiltonian(N, J, Xi):
    """
    Construct the SYK Hamiltonian for a given number of Majorana fermions.

    Parameters:
        N (int): Number of Majorana fermions (must be even and greater than 4).
        J (float): Disorder strength.
        Xi (ndarray): Majorana fermion representation.

    Returns:
        H (ndarray): The SYK Hamiltonian.
    """
    if N % 2 != 0 or N < 4:
        raise ValueError("N must be even and greater than 4.")

    H_length = 2 ** (N // 2)
    H = np.zeros((H_length, H_length), dtype=complex)

    sigma = np.sqrt(np.math.factorial(3) * J ** 2 / ((N - 1) * (N - 2) * (N - 3)))

    for j in range(1, N + 1):
        for k in range(j + 1, N + 1):
            for l in range(k + 1, N + 1):
                for m in range(l + 1, N + 1):
                    J_jklm = np.random.randn() * sigma
                    H += J_jklm * np.dot(np.dot(np.dot(Xi[:, :, j - 1], Xi[:, :, k - 1]), Xi[:, :, l - 1]), Xi[:, :, m - 1])

    return H

def syk_hamiltonian_q3(N, J, Xi):
    """
    Construct the q=3 SYK Hamiltonian.

    Parameters:
        N (int): Number of Majorana fermions (must be even and greater than 4).
        J (float): Disorder strength.
        Xi (ndarray): Majorana fermion representation.

    Returns:
        H (ndarray): The q=3 SYK Hamiltonian.
    """
    if N % 2 != 0 or N < 4:
        raise ValueError("N must be even and greater than 4.")

    H_length = 2 ** (N // 2)
    H = np.zeros((H_length, H_length), dtype=complex)

    sigma = np.sqrt(np.math.factorial(3) * J ** 2 / ((N - 1) * (N - 2) * (N - 3)))

    for j in range(1, N + 1):
        for k in range(j + 1, N + 1):
            for l in range(k + 1, N + 1):
                for m in range(l + 1, N + 1):
                    for p in range(m + 1, N + 1):
                        for q in range(p + 1, N + 1):
                            J_jklmpq = np.random.randn() * sigma
                            H += J_jklmpq * np.dot(np.dot(np.dot(np.dot(np.dot(Xi[:, :, j - 1], Xi[:, :, k - 1]),
                                                                          Xi[:, :, l - 1]), Xi[:, :, m - 1]),
                                                         Xi[:, :, p - 1]), Xi[:, :, q - 1])

    return H

# Load data to Google Colab
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

save_path = '/content/drive/MyDrive/Data/data.zip'
N = 16  # Number of Majorana fermions (must be even for SYK)
disorder_strength = 1
num_samples = 100
Xi = majorana(N)

def haar_random_state(dimension):
    """
    Generate a Haar random state (normalized complex vector).

    Parameters:
        dimension (int): Dimension of the Hilbert space.

    Returns:
        normalized_vector (ndarray): Normalized random complex vector.
    """
    random_vector = np.random.normal(0, 1, size=(dimension,))
    return random_vector / np.linalg.norm(random_vector)

# Generate training data
hilbert_space_dimension = 2 ** (N // 2)

def my_gen(N_sample):
    """
    Generate SYK and Haar random states.

    Parameters:
        N_sample (int): Number of samples to generate.

    Returns:
        pd.DataFrame: Concatenated DataFrame of generated data.
    """
    X = np.empty((N_sample, hilbert_space_dimension + 1))
    y = np.zeros((N_sample, hilbert_space_dimension + 1))
    z = np.zeros((N_sample * 2, hilbert_space_dimension + 1))

    for k in range(N_sample):
        if k < N_sample:
            X[k, :hilbert_space_dimension] = haar_random_state(hilbert_space_dimension)
            z[k + N_sample, :] = X[k, :]
        H = syk_hamiltonian(N, disorder_strength, Xi)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        y[k, :hilbert_space_dimension] = eigenvectors[0] / np.linalg.norm(eigenvectors[0])
        z[k, :] = y[k, :]

    column_names = [f'C_WF_{i}' for i in range(hilbert_space_dimension)]
    z1 = pd.DataFrame(z[:, :hilbert_space_dimension], columns=column_names)
    z2 = pd.DataFrame(z[:, hilbert_space_dimension] > 0.1, columns=['is SYK'])
    return pd.concat([z1, z2], axis=1)

def my_data_gen(N, n_jobs=-1):
    """
    Generate a DataFrame with SYK and Haar random states.

    Parameters:
        N (int): Number of samples.
        n_jobs (int): Number of processors for parallel computing (-1 for maximum).

    Returns:
        pd.DataFrame: DataFrame containing generated data.
    """
    t = time.time()
    n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
    print(f'Running with {n_jobs} jobs')

    with mp.Pool(n_jobs) as pool:
        results = pool.map(my_gen, [N // n_jobs for _ in range(n_jobs)])

    print(f"Time taken: {time.time() - t} seconds")
    return pd.concat(results, ignore_index=True)

# Generate data
data_frame = my_data_gen(num_samples, n_jobs=-1)

# Save data to .csv format
data_frame.to_csv('training_data.csv', index=False)

# Load data
training_data = pd.read_csv('training_data.csv')
print(training_data.head())
