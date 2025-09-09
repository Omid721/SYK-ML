"""
This script implements an unsupervised neural network (autoencoder) to learn the features of the SYK ground state.
The model is trained on a dataset of SYK ground states.
The performance of the model is evaluated by its ability to reconstruct the original wave functions.
"""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Suppress warnings
warnings.filterwarnings("ignore")


def load_data(file_path):
    """Load data from a specified CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)


def preprocess_data(data):
    """Preprocess the data for training and testing.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        tuple: A tuple containing the training and testing data.
    """
    # Filter for SYK ground states
    data = data[data["is_SYK"] == True]

    # Shuffle the data
    data = shuffle(data, random_state=0)

    # Define the features
    features = data.columns.drop("is_SYK")

    # Split the data into training and testing sets
    X = data[features]
    X_train, X_test, _, _ = train_test_split(
        X, X, test_size=0.2, random_state=42
    )

    return X_train, X_test, features


def build_autoencoder(input_shape, N):
    """Build an autoencoder model.

    Args:
        input_shape (int): Number of input features.
        N (int): Number of Majorana fermions.

    Returns:
        tf.keras.Sequential: Compiled autoencoder model.
    """
    # The size of the hidden layer is chosen based on the formula for the number of SYK couplings
    hidden_size = int(
        1 / 24 * (-6 * N + 11 * N ** 2 - 6 * N ** 3 + N ** 4)
    )

    # Encoder
    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_shape,)),
            tf.keras.layers.Dense(hidden_size, activation="tanh"),
        ]
    )

    # Decoder
    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(hidden_size,)),
            tf.keras.layers.Dense(input_shape, activation="tanh"),
        ]
    )

    # Combine encoder and decoder to create autoencoder
    autoencoder = tf.keras.Sequential([encoder, decoder])

    return autoencoder


def train_model(model, X_train, epochs=50, batch_size=32):
    """Train the autoencoder model.

    Args:
        model (tf.keras.Sequential): The autoencoder model.
        X_train (pd.DataFrame): The training data.
        epochs (int): The number of epochs to train for.
        batch_size (int): The batch size for training.
    """
    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=0)


def plot_reconstructions(original, reconstructed, n=5):
    """Plot the original and reconstructed wave functions.

    Args:
        original (pd.DataFrame): The original wave functions.
        reconstructed (np.ndarray): The reconstructed wave functions.
        n (int): The number of wave functions to plot.
    """
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original.iloc[i].values.reshape(int(np.sqrt(len(original.iloc[i]))),-1), cmap="viridis")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title("Original Wave Functions")

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(int(np.sqrt(len(reconstructed[i]))),-1), cmap="viridis")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title("Reconstructed Wave Functions")

    plt.show()


def main():
    """
    Main function to run the unsupervised learning experiment.
    """
    # Define the file paths for the datasets
    N_values = [8]
    file_paths = {f"N{N}": f"SYK_data_N{N}.csv" for N in N_values}

    # Iterate over the datasets
    for key, path in file_paths.items():
        print(f"Processing {key} data...")
        N = int(key.replace("N", ""))

        # Load and preprocess the data
        data = load_data(path)
        X_train, X_test, features = preprocess_data(data)

        # Build, train, and evaluate the model
        autoencoder = build_autoencoder(len(features), N)
        train_model(autoencoder, X_train)

        # Reconstruct the test data
        reconstructed_wf = autoencoder.predict(X_test)

        # Plot the original and reconstructed wave functions
        plot_reconstructions(X_test, reconstructed_wf)


if __name__ == "__main__":
    main()