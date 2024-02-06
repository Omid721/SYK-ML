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


#################################################################################



from google.colab import drive
import zipfile
import io

# Mount Google Drive
drive.mount('/content/drive')


# Path to the uploaded zip file
#zip_file_path = 'https://drive.google.com/file/d/1-0cSWR6b8uSBCFI3vSHNWaixX5S0r_A7/view?usp=share_link'

zip_file_path = '/content/drive/MyDrive/Data/data_SYK_N6.zip'
zip_file_path2 = '/content/drive/MyDrive/Data/data_SYK_N8.zip'
zip_file_path3 = '/content/drive/MyDrive/Data/data_SYK_N10.zip'
zip_file_path4 = '/content/drive/MyDrive/Data/data_SYK_N12.zip'

#zip_file_path = '/content/drive/MyDrive/Data/data.zip'

file_to_read = 'data'

# Read the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as archive:
    # Extract the specific file from the ZIP archive
    with archive.open(file_to_read) as file:
        # Read the CSV file
        data_N6 = pd.read_csv(file)

with zipfile.ZipFile(zip_file_path2, 'r') as archive:
    # Extract the specific file from the ZIP archive
    with archive.open(file_to_read) as file:
        # Read the CSV file
        data_N8 = pd.read_csv(file)

with zipfile.ZipFile(zip_file_path3, 'r') as archive:
    # Extract the specific file from the ZIP archive
    with archive.open(file_to_read) as file:
        # Read the CSV file
        data_N10 = pd.read_csv(file)

with zipfile.ZipFile(zip_file_path4, 'r') as archive:
    # Extract the specific file from the ZIP archive
    with archive.open(file_to_read) as file:
        # Read the CSV file
        data_N12 = pd.read_csv(file)




data_N6 = data_N6[data_N6['is SYK'] == True]
data_N8 = data_N8[data_N8['is SYK'] == True]
data_N10 = data_N10[data_N10['is SYK'] == True]
data_N12 = data_N12[data_N12['is SYK'] == True]

###########################################################################################################


test_size = 0.05

data_N6 = sk.utils.shuffle(data_N6,random_state=0)
target = 'is SYK'
features_N6 = data_N6.columns.drop([target,'Unnamed: 0'])
y_N6 = data_N6[target].astype(int)
x_N6 = data_N6[features_N6]
x_train_N6, x_test_N6 , y_train_N6, y_test_N6 = sk.model_selection.train_test_split(x_N6 , x_N6 , random_state=0,test_size=test_size)



data_N8 = sk.utils.shuffle(data_N8,random_state=0)
features_N8 = data_N8.columns.drop([target, 'Unnamed: 0'])
y_N8 = data_N8[target].astype(int)
x_N8 = data_N8[features_N8]
x_train_N8, x_test_N8 , y_train_N8, y_test_N8 = sk.model_selection.train_test_split(x_N8 , x_N8 , random_state=0,test_size=test_size)



data_N10 = sk.utils.shuffle(data_N10,random_state=0)
features_N10 = data_N10.columns.drop([target, 'Unnamed: 0'])
y_N10 = data_N10[target].astype(int)
x_N10 = data_N10[features_N10]
x_train_N10, x_test_N10 , y_train_N10, y_test_N10 = sk.model_selection.train_test_split(x_N10 , x_N10 , random_state=0,test_size=test_size)


data_N12 = sk.utils.shuffle(data_N12,random_state=0)
features_N12 = data_N12.columns.drop([target, 'Unnamed: 0'])
y_N12 = data_N12[target].astype(int)
x_N12 = data_N12[features_N12]
x_train_N12, x_test_N12 , y_train_N12, y_test_N12 = sk.model_selection.train_test_split(x_N12 , x_N12 , random_state=0,test_size=test_size)


###########################################################################################################





def build_nn_model(features_N, N):
    input_size = len(features_N)  # Number of input features
    hidden_size = int(1/24 *(-6*N + 11*N**2 - 6*N**3 + N**4))  # Size of the hidden layer
    output_size = input_size
    # Encoder
    encoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(input_size,)), tf.keras.layers.Dense(hidden_size, activation='tanh')])

# Decoder
    decoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(hidden_size,)), tf.keras.layers.Dense(input_size, activation='tanh')])

# Combine encoder and decoder to create autoencoder
    autoencoder = tf.keras.Sequential([encoder, decoder])

    model =  autoencoder
    return model

# Compile the autoencoder

model_N6 = build_nn_model(features_N6, 6)
model_N8 = build_nn_model(features_N8, 8)
model_N10 = build_nn_model(features_N10, 10)
model_N12 = build_nn_model(features_N12, 12)

model_N6.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
model_N8.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
model_N10.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
model_N12.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
model_N6.fit(x_train_N6, y_train_N6, epochs=50, batch_size=32)
model_N8.fit(x_train_N8, y_train_N8, epochs=50, batch_size=32)
model_N10.fit(x_train_N10, y_train_N10, epochs=50, batch_size=32)
model_N12.fit(x_train_N12, y_train_N12, epochs=50, batch_size=32)


# Now you can train the autoencoder using your data


####################################################################################

# Evaluate the model on the test set
loss_N6, accuracy_N6 = model_N6.evaluate(x_test_N6, y_test_N6)
loss_N8, accuracy_N8 = model_N8.evaluate(x_test_N8, y_test_N8)
loss_N10, accuracy_N10 = model_N10.evaluate(x_test_N10, y_test_N10)
loss_N12, accuracy_N12 = model_N12.evaluate(x_test_N12, y_test_N12)

loss = np.array([loss_N6, loss_N8, loss_N10, loss_N12])
accuracy = np.array([accuracy_N6, accuracy_N8, accuracy_N10, accuracy_N12])

print("loss is:", loss)
print("accuracy is", accuracy)
print("="*80)
# Predict the ground state using the trained neural network
#decision_values_NN = model.predict(x_test)
#our_case_prediction = model.predict(x_test_our_case)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot Loss
ax1.bar(['N6', 'N8', 'N10', 'N12'], loss, color='red')
ax1.set_title('Loss')
ax1.set_ylabel('Loss Value')

# Plot Accuracy
ax2.bar(['N6', 'N8', 'N10', 'N12'], accuracy, color='blue')
ax2.set_title('Accuracy')
ax2.set_ylabel('Accuracy Value')

plt.show()


#################################################################################








import tensorflow as tf
import numpy as np

# Define the Pauli matrices
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

# Define the Sachdev-Ye-Kitaev (SYK) Hamiltonian
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
    J = np.zeros(N**4)
    o=0
    # Iterate over combinations (j, k, l, m)
    for j in range(1, N+1):
        for k in range(j+1, N+1):
            for l in range(k+1, N+1):
                for m in range(l+1, N+1):
                    J_jklm = np.random.randn() * sigma
                    J[o] = J_jklm
                    o=o+1
                    H += J_jklm * np.dot(np.dot(np.dot(Xi[:, :, j-1], Xi[:, :, k-1]), Xi[:, :, l-1]), Xi[:, :, m-1])

    return H, J




###############################################################

import pandas as pd
from google.colab import drive
# Generate testing data
def my_gen(N_sample, N):
  Xi = majorana(N)
  hilbert_space_dimension = 2**(N//2)
  disorder_strength = 1
  N_SYK_sample = N_sample
  y = np.zeros((N_SYK_sample, hilbert_space_dimension+1))

  for k in range(N_SYK_sample):
    H, J = syk_hamiltonian(N, disorder_strength, Xi)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    y[k, 0:hilbert_space_dimension] = eigenvectors[0]/np.linalg.norm(eigenvectors[0])
    y[k,hilbert_space_dimension] = 1
      #print(k)
      #print("="*80)

  column_names = [f'C_WF_{i}' for i in range(hilbert_space_dimension)]
  z1 = y[:,0:hilbert_space_dimension]
  z2 = y[:,hilbert_space_dimension]
  z1 = pd.DataFrame(z1 , columns = column_names)
  z2 = pd.DataFrame(z2 > 0.1, columns = ['is SYK'])
  return pd.concat([z1,z2],axis=1) , J
  #return X, y, z


def preprocess_data(WF, J, features):
    WF = WF[WF['is SYK']==True]
    features = WF.columns.drop(['is SYK'])
    WF = WF[features]
    J = np.sort(J[J != 0])
    return WF, J

data = []

Ns =[6,8,10,12]
N_list = [f"N{i}" for i in Ns]
WF = {n :[] for n in N_list}
Js = {n :[] for n in N_list}
features = {"N6":features_N6 ,"N8":features_N8, "N10":features_N10, "N12":features_N12}

for n in Ns:
    WF[f"N{n}"], Js[f"N{n}"] = my_gen(1, n) #use my_gen
    WF[f"N{n}"], Js[f"N{n}"] = preprocess_data(WF[f"N{n}"], Js[f"N{n}"], features[f"N{n}"])
    data.append((WF[f"N{n}"], Js[f"N{n}"]))

for i, n in enumerate(N_list):
  WF[n], Js[n] = data[i]


print(len(Js["N10"]))



###############################################################################################

encoder_N8 = model_N8.layers[0]

J = Js['N8']

#print(non_zero_elements)
J_predict = encoder_N8.predict(WF['N8'])

J = np.sort(J[J != 0])
J_predict = np.sort(J_predict)


# Ensure both arrays have the same length
#min_length = min(len(J), len(J_predict))
#J = J[:min_length]
#J_predict = J_predict[:min_length]


plt.figure(figsize=(8, 6))
plt.scatter(J, J_predict, color='blue', alpha=0.5)
plt.scatter(J, J, color='red')  # Plot y = x line for reference
plt.plot(J, J, color='red', linestyle = '--')  # Plot y = x line for reference
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values (J)')
plt.ylabel('Predicted Values (J_predict)')
plt.grid(True)
plt.show()


##########################################
encoded_samples = encoder_N8.predict(WF['N8'])
decoder_N8 = model_N8.layers[1]
decoded_samples = decoder_N8.predict(encoded_samples)



# Visualize original and reconstructed samples

plt.subplot(2, 5, i + 1)
plt.imshow((WF['N8'].values).reshape(4, 4), cmap='viridis')
plt.title('Original_WF')
plt.axis('off')

plt.subplot(2, 5, i + 6)
plt.imshow(decoded_samples.reshape(4, 4), cmap='viridis')
plt.title('Recons_WF')
plt.axis('off')

plt.show()

##########################################################

#for N=12
encoder_N12 = model_N12.layers[0]

J = Js['N12']

#print(non_zero_elements)
J_predict = encoder_N12.predict(WF['N12'])

J = np.sort(J[J != 0])
J_predict = np.sort(J_predict)


# Ensure both arrays have the same length
#min_length = min(len(J), len(J_predict))
#J = J[:min_length]
#J_predict = J_predict[:min_length]


plt.figure(figsize=(8, 6))
plt.scatter(J, J_predict, color='blue', alpha=0.5)
plt.scatter(J, J, color='red')  # Plot y = x line for reference
plt.plot(J, J, color='red', linestyle = '--')  # Plot y = x line for reference
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values (J)')
plt.ylabel('Predicted Values (J_predict)')
plt.grid(True)
plt.show()


encoded_samples = encoder_N12.predict(WF['N12'])
decoder_N12 = model_N12.layers[1]
decoded_samples = decoder_N12.predict(encoded_samples)



# Visualize original and reconstructed samples
i=0
plt.subplot(2, 5, i + 1)
plt.imshow((WF['N12'].values).reshape(8, 8), cmap='viridis')
plt.title('Original_WF')
plt.axis('off')

plt.subplot(2, 5, i + 6)
plt.imshow(decoded_samples.reshape(8, 8), cmap='viridis')
plt.title('Recons_WF')
plt.axis('off')

plt.show()

###################################################################################
#N=8 WF check

encoded_samples = encoder_N8.predict(x_test_N8[:5])
decoder_N8 = model_N8.layers[1]
decoded_samples = decoder_N8.predict(encoded_samples)



# Visualize original and reconstructed samples
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow((x_test_N8.iloc[i].values).reshape(4, 4), cmap='viridis')
    plt.title('Original_WF')
    plt.axis('off')

    plt.subplot(2, 5, i + 6)
    plt.imshow(decoded_samples[i].reshape(4, 4), cmap='viridis')
    plt.title('Recons_WF')
    plt.axis('off')

plt.show()

#######################################################
#N=12 WF Check

encoded_samples = encoder_N12.predict(x_test_N12[:5])
decoder_N12 = model_N12.layers[1]
decoded_samples = decoder_N12.predict(encoded_samples)



# Visualize original and reconstructed samples
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow((x_test_N12.iloc[i].values).reshape(8, 8), cmap='viridis')
    plt.title('Original_WF')
    plt.axis('off')

    plt.subplot(2, 5, i + 6)
    plt.imshow(decoded_samples[i].reshape(8, 8), cmap='viridis')
    plt.title('Recons_WF')
    plt.axis('off')

plt.show()
