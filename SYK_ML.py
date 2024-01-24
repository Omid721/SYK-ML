import time
##################
import warnings
import pickle
import io, requests, zipfile
import contextlib # redirect_stdout
import sys

#####
import multiprocessing as mp # Pool , cpu_count
import numpy as np
import scipy as sp
import pandas as pd
import sympy as sym
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
import sklearn as sk #submodules does not import automatically

#from numba import jit, prange
###############################################################
import scipy.optimize # minimize
import scipy.stats # expon
#####
import sklearn.utils
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
#####
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap, SpectralEmbedding, TSNE
#####
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
#####
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

#####################################################################

# Define the Pauli matrices
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

# Define the Sachdev-Ye-Kitaev (SYK) Hamiltonian
def syk_hamiltonian(N, disorder_strength):
    # Define the number of Majorana fermions (N should be even)
    if N % 2 != 0:
        raise ValueError("Number of Majorana fermions (N) should be even.")

    # Random couplings
    J = disorder_strength * np.random.randn(N, N, N, N) / np.sqrt(2)

    # Build the Hamiltonian
    H = np.zeros((2**N, 2**N), dtype=np.complex128)

    for i in range(N):
        for j in range(i+1,N):
            for k in range(j+1,N):
                for l in range(k+1,N):
                    H += J[i, j, k, l] * 1j * np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(np.eye(2**(i)), sigma_y), np.eye(2**(j-i-1))), sigma_y), np.eye(2**(k-j-1))) ,sigma_y), np.eye(2**(l-k-1))), sigma_y), np.eye(2**(N-l-1)))

    return H



    ###############################################################

#from google.colab import drive
#drive.mount('/content/drive')
#save_path = '/content/drive/MyDrive/Data/data.zip'
# Generate training data
N = 6 # Number of Majorana fermions (even for SYK)
disorder_strength = 1
num_samples = 100

def haar_random_state(dimension):
    # Generate a random complex vector
    random_vector = np.random.normal(0, 1, size=(dimension,))
    # Normalize the vector
    normalized_vector = random_vector / np.linalg.norm(random_vector)

    return normalized_vector

# Set the dimension of the Hilbert space
hilbert_space_dimension = 2**N


# Generate a Haar random state



def my_gen(N_SYK_sample , N_Haar_sample):
  X = np.empty((N_Haar_sample, hilbert_space_dimension+1))
  y = np.zeros((N_SYK_sample, hilbert_space_dimension+1))
  z = np.zeros((N_SYK_sample+N_Haar_sample, hilbert_space_dimension+1))

  for k in range(max(N_SYK_sample,N_Haar_sample)):
    if k < N_Haar_sample :
      X[k, 0:hilbert_space_dimension] = haar_random_state(hilbert_space_dimension)
      X[k,hilbert_space_dimension] = 0
      z[k+ N_SYK_sample, :] = X[k,:]
    if k < N_SYK_sample:
      H = syk_hamiltonian(N, disorder_strength)
      eigenvalues, eigenvectors = np.linalg.eigh(H)
      y[k, 0:hilbert_space_dimension] = eigenvectors[0]/np.linalg.norm(eigenvectors[0])
      y[k,hilbert_space_dimension] = 1
      z[k,:] = y[k,:]

  column_names = [f'C_WF_{i}' for i in range(hilbert_space_dimension)]
  z1 = z[:,0:hilbert_space_dimension]
  z2 = z[:,hilbert_space_dimension]
  z1 = pd.DataFrame(z1 , columns = column_names)
  z2 = pd.DataFrame(z2 > 0.1, columns = ['is SYK'])
  return pd.concat([z1,z2],axis=1)
  #return X, y, z

#data = my_gen(100,50)
#data.to_csv(save_path, index=True)



################################################################

# Sample data generation
data = my_gen(500, 500)
data = sk.utils.shuffle(data,random_state=0)
target = 'is SYK'
features = data.columns.drop(['is SYK'])
y_train = data[target].astype(int)
x_train = data[features]


#########################################################################


# Define the neural network model with its layers and activation function
def build_nn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Change activation to 'sigmoid' for binary classification
    ])
    return model

# Build and compile the neural network model with the correct loss function and optimizer
model = build_nn_model()

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Display model summary
model.summary()


#########################################################
def test_gen(a, N):
  X = np.empty((1,hilbert_space_dimension), dtype = complex)
  psi = haar_random_state(hilbert_space_dimension)
  H = syk_hamiltonian(N, disorder_strength)
  for i in range(N):
    X[0,:] += (1/np.math.factorial(i))*(a*H)**i @ psi
  X_real = X.real.astype(float)
  column_names = [f'C_WF_{i}' for i in range(hilbert_space_dimension)]
  X = pd.DataFrame(X_real , columns = column_names)
  return X

  #################################################################

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
print("="*80)
print("Predicted probabilities for being SYK:", our_case_prediction)


#############################################################

from google.colab import drive
import zipfile
import io

# Mount Google Drive
drive.mount('/content/drive')

# Path to the uploaded zip file
zip_file_path = '/content/drive/MyDrive/Data/data_N_10.zip'

file_to_read = 'data'

# Read the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as archive:
    # Extract the specific file from the ZIP archive
    with archive.open(file_to_read) as file:
        # Read the CSV file
        data = pd.read_csv(file)

# Now 'data' contains the DataFrame with the content of the specified CSV file
print(data)
print(data.describe)
