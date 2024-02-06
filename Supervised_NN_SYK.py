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

#loading the data

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

####################################################################

#building the NN model and train it based on the Haar random states and GS of SYK for N=6,...,12

test_size = 0.05

data_N6 = sk.utils.shuffle(data_N6,random_state=0)
target = 'is SYK'
features_N6 = data_N6.columns.drop([target,'Unnamed: 0'])
y_N6 = data_N6[target].astype(int)
x_N6 = data_N6[features_N6]
x_train_N6, x_test_N6 , y_train_N6, y_test_N6 = sk.model_selection.train_test_split(x_N6 , y_N6 , random_state=0,test_size=test_size)



data_N8 = sk.utils.shuffle(data_N8,random_state=0)
features_N8 = data_N8.columns.drop([target, 'Unnamed: 0'])
y_N8 = data_N8[target].astype(int)
x_N8 = data_N8[features_N8]
x_train_N8, x_test_N8 , y_train_N8, y_test_N8 = sk.model_selection.train_test_split(x_N8 , y_N8 , random_state=0,test_size=test_size)



data_N10 = sk.utils.shuffle(data_N10,random_state=0)
features_N10 = data_N10.columns.drop([target, 'Unnamed: 0'])
y_N10 = data_N10[target].astype(int)
x_N10 = data_N10[features_N10]
x_train_N10, x_test_N10 , y_train_N10, y_test_N10 = sk.model_selection.train_test_split(x_N10 , y_N10 , random_state=0,test_size=test_size)


data_N12 = sk.utils.shuffle(data_N12,random_state=0)
features_N12 = data_N12.columns.drop([target, 'Unnamed: 0'])
y_N12 = data_N12[target].astype(int)
x_N12 = data_N12[features_N12]
x_train_N12, x_test_N12 , y_train_N12, y_test_N12 = sk.model_selection.train_test_split(x_N12 , y_N12 , random_state=0,test_size=test_size)

def build_nn_model(features_N):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features_N),)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Change activation to 'sigmoid' for binary classification
    ])
    return model

# Build and compile the neural network model with the correct loss function and optimizer
model_N6 = build_nn_model(features_N6)
model_N8 = build_nn_model(features_N8)
model_N10 = build_nn_model(features_N10)
model_N12 = build_nn_model(features_N12)

model_N6.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
model_N8.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
model_N10.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
model_N12.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
model_N6.fit(x_train_N6, y_train_N6, epochs=50, batch_size=32)
model_N8.fit(x_train_N8, y_train_N8, epochs=50, batch_size=32)
model_N10.fit(x_train_N10, y_train_N10, epochs=50, batch_size=32)
model_N12.fit(x_train_N12, y_train_N12, epochs=50, batch_size=32)

# Display model summary
print("N = 6 model summary :")
print(model_N6.summary())
print("---"*20)
print("N = 8 model summary :")
print(model_N8.summary())
print("---"*20)
print("N = 10 model summary :")
print(model_N10.summary())
print("---"*20)
print("N = 12 model summary :")
print(model_N12.summary())
print("---"*20)

######################################################################

#check the loss and accuracy VS N

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
