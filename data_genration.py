%time
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


######################################################################################################

#SYK_Hamiltonian
import tensorflow as tf
import numpy as np

# Define the Pauli matrices
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

# Define the Sachdev-Ye-Kitaev (SYK) Hamiltonian
'''
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
'''

###################################################################################

from numpy import identity

def tensor(a, i, N, p=2):
    """
    TENSOR takes matrix a and implements it as the ith qubit
    Embeds the matrix a in a p^N dimensional Hilbert space of N qudits
    in the ith position in the tensor product.
    It is assumed a is a pxp 1-qudit matrix.

    Parameters:
    - a: pxp Unitary matrix to operate on a single qudit
    - i: Position in the tensor product, indexing from 1
    - N: Number of qudits in the Hilbert space
    - p: Degree of freedom per site (default is 2 for qubits)

    Returns:
    - A: Resulting matrix after tensor product
    """
    dim1 = int(p**(i-1))
    dim2 = int(p**(N-i))

    A = np.kron(np.kron(np.identity(dim1), a), np.identity(dim2))
    return A

def majorana(N):
    # MAJORANA creates a representation of N majorana fermions

    if N % 2 != 0:
        raise ValueError("N must be even")

    # SU(2) Matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Output of majoranas here
    dim = int(2**(N/2))
    Xi = np.zeros((dim, dim, N), dtype=complex)

    # Growing chain of Zs, start with 0
    Zs = np.eye(dim, dtype=complex)

    for i in range(1, N+1):
        # The X or Y at the end
        if i % 2 == 1:
            xi = tensor(X, (i+1)//2, N//2)
        else:
            xi = tensor(Y, i//2, N//2)

        Xi[:, :, i-1] = np.dot(Zs, xi)

        # Build an Z string
        if i % 2 == 0:
            Zs = np.dot(Zs, tensor(Z, i//2, N//2))

    #Kitaev's normalization
    Xi = Xi / np.sqrt(2)


    return Xi

def syk_hamiltonian(N, J, Xi):

    if N % 2 != 0:
        raise ValueError("N must be even")

    if N < 4:
        raise ValueError("N must be greater than 4")

    # build Hamiltonian
    H_length = 2**(N//2)
    H = np.zeros((H_length, H_length), dtype=complex)

    var_J_jklm = np.math.factorial(3) * J**2 / ((N-1)*(N-2)*(N-3))
    sigma = np.sqrt(var_J_jklm)

    # Iterate over combinations (j, k, l, m)
    for j in range(1, N+1):
        for k in range(j+1, N+1):
            for l in range(k+1, N+1):
                for m in range(l+1, N+1):
                    J_jklm = np.random.randn() * sigma
                    H += J_jklm * np.dot(np.dot(np.dot(Xi[:, :, j-1], Xi[:, :, k-1]), Xi[:, :, l-1]), Xi[:, :, m-1])

    return H


# This Hamiltonian is q=3 SYK hamiltonian and is made to check if we can make the q=3 GS by our model
def syk_hamiltonian_q3(N, J, Xi):

    if N % 2 != 0:
        raise ValueError("N must be even")

    if N < 4:
        raise ValueError("N must be greater than 4")

    # build Hamiltonian
    H_length = 2**(N//2)
    H = np.zeros((H_length, H_length), dtype=complex)

    var_J_jklm = np.math.factorial(3) * J**2 / ((N-1)*(N-2)*(N-3))
    sigma = np.sqrt(var_J_jklm)

    # Iterate over combinations (j, k, l, m)
    for j in range(1, N+1):
        for k in range(j+1, N+1):
            for l in range(k+1, N+1):
                for m in range(l+1, N+1):
                  for p in range(m+1, N+1):
                    for q in range(p+1, N+1):
                      J_jklmpq = np.random.randn() * sigma
                      H += J_jklmpq * np.dot(np.dot(np.dot(np.dot(np.dot(Xi[:, :, j-1], Xi[:, :, k-1]), Xi[:, :, l-1]), Xi[:, :, m-1]), Xi[:,:,p-1]), Xi[:,:,q-1])

    return H




# run this in google colab for N = 6, 8, 10, 12, 14, ...

import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
save_path = '/content/drive/MyDrive/Data/data.zip'
# Generate training data
N = 16  # Number of Majorana fermions (even for SYK)
disorder_strength = 1
num_samples = 100
Xi = majorana(N)

def haar_random_state(dimension):
    # Generate a random complex vector
    random_vector = np.random.normal(0, 1, size=(dimension,))
    # Normalize the vector
    normalized_vector = random_vector / np.linalg.norm(random_vector)

    return normalized_vector

# Set the dimension of the Hilbert space
hilbert_space_dimension = 2**(N//2)


# Generate a Haar random state



def my_gen(N_sample):
  N_SYK_sample = N_sample
  N_Haar_sample = N_sample
  X = np.empty((N_Haar_sample, hilbert_space_dimension+1))
  y = np.zeros((N_SYK_sample, hilbert_space_dimension+1))
  z = np.zeros((N_SYK_sample+N_Haar_sample, hilbert_space_dimension+1))

  for k in range(max(N_SYK_sample,N_Haar_sample)):
    if k < N_Haar_sample :
      X[k, 0:hilbert_space_dimension] = haar_random_state(hilbert_space_dimension)
      X[k,hilbert_space_dimension] = 0
      z[k+ N_SYK_sample, :] = X[k,:]
    if k < N_SYK_sample:
      H = syk_hamiltonian(N, disorder_strength, Xi)
      eigenvalues, eigenvectors = np.linalg.eigh(H)
      y[k, 0:hilbert_space_dimension] = eigenvectors[0]/np.linalg.norm(eigenvectors[0])
      y[k,hilbert_space_dimension] = 1
      z[k,:] = y[k,:]
      #print(k)
      #print("="*80)

  column_names = [f'C_WF_{i}' for i in range(hilbert_space_dimension)]
  z1 = z[:,0:hilbert_space_dimension]
  z2 = z[:,hilbert_space_dimension]
  z1 = pd.DataFrame(z1 , columns = column_names)
  z2 = pd.DataFrame(z2 > 0.1, columns = ['is SYK'])
  return pd.concat([z1,z2],axis=1)
  #return X, y, z



def my_data_gen(N, n_jobs=-1):
  '''
  N : int : number of samples : O(N)
  n_jobs : int : number of processors for parallel computing, -1  for max possible

  --- return ---
  data : pd.DataFrame : data
  '''

  t=time.time()
  if n_jobs == -1: n_jobs = mp.cpu_count()
  print('running with n_jobs : {}'.format(n_jobs))
  f=n_jobs*100
  N_list = [N//f]*f + [N%f]

  data_list=[]
  with mp.Pool(processes=n_jobs) as pool:
      for i,data in enumerate(pool.imap_unordered(my_gen,N_list) , 1):
          elapsed_time = int(time.time()-t)
          sys.stderr.write('\rdone {:.2%} , time : {} min , {} s , estimated total time : {} min , {} s '.format( i/len(N_list) , elapsed_time//60 , elapsed_time%60 , int(len(N_list)/i * elapsed_time)//60 , int(len(N_list)/i * elapsed_time)%60 ))
          #print(f"Iteration {i}")
          data_list += [data]

  data=pd.concat(data_list)
  return data

data = my_data_gen(1000)
data.to_csv(save_path, index=True)

###########################################################
