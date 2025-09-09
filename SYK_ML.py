"""
This script provides a command-line interface to run the supervised and unsupervised learning tasks for the SYK_ML project.
"""

import argparse
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


# ----------------- Supervised Learning Functions ----------------- #

def preprocess_supervised_data(data):
    """Preprocess the data for supervised learning.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        tuple: A tuple containing the training and testing data.
    """
    data = shuffle(data, random_state=0)
    target = "is_SYK"
    features = data.columns.drop(target)
    X = data[features]
    y = data[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, features


def build_classifier(input_shape):
    """Build a neural network model for binary classification.

    Args:
        input_shape (int): Number of input features.

    Returns:
        tf.keras.Sequential: Compiled neural network model.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                64, activation="relu", input_shape=(input_shape,)
            ),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


def run_supervised_task(N):
    """Run the supervised learning task for a given N.

    Args:
        N (int): The number of Majorana fermions.
    """
    print(f"Running supervised task for N={N}...")
    file_path = f"SYK_data_N{N}.csv"
    data = load_data(file_path)
    X_train, X_test, y_train, y_test, features = preprocess_supervised_data(data)

    model = build_classifier(len(features))
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


# ----------------- Unsupervised Learning Functions ----------------- #

def preprocess_unsupervised_data(data):
    """Preprocess the data for unsupervised learning.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        tuple: A tuple containing the training and testing data.
    """
    data = data[data["is_SYK"] == True]
    data = shuffle(data, random_state=0)
    features = data.columns.drop("is_SYK")
    X = data[features]
    X_train, X_test, _, _ = train_test_split(X, X, test_size=0.2, random_state=42)
    return X_train, X_test, features


def build_autoencoder(input_shape, N):
    """Build an autoencoder model.

    Args:
        input_shape (int): Number of input features.
        N (int): Number of Majorana fermions.

    Returns:
        tf.keras.Sequential: Compiled autoencoder model.
    """
    hidden_size = int(1 / 24 * (-6 * N + 11 * N ** 2 - 6 * N ** 3 + N ** 4))
    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_shape,)),
            tf.keras.layers.Dense(hidden_size, activation="tanh"),
        ]
    )
    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(hidden_size,)),
            tf.keras.layers.Dense(input_shape, activation="tanh"),
        ]
    )
    autoencoder = tf.keras.Sequential([encoder, decoder])
    autoencoder.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return autoencoder


def plot_reconstructions(original, reconstructed, n=5):
    """Plot the original and reconstructed wave functions.

    Args:
        original (pd.DataFrame): The original wave functions.
        reconstructed (np.ndarray): The reconstructed wave functions.
        n (int): The number of wave functions to plot.
    """
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original.iloc[i].values.reshape(int(np.sqrt(len(original.iloc[i]))),-1), cmap="viridis")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title("Original Wave Functions")

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(int(np.sqrt(len(reconstructed[i]))),-1), cmap="viridis")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title("Reconstructed Wave Functions")

    plt.show()


def run_unsupervised_task(N):
    """Run the unsupervised learning task for a given N.

    Args:
        N (int): The number of Majorana fermions.
    """
    print(f"Running unsupervised task for N={N}...")
    file_path = f"SYK_data_N{N}.csv"
    data = load_data(file_path)
    X_train, X_test, features = preprocess_unsupervised_data(data)

    autoencoder = build_autoencoder(len(features), N)
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)

    reconstructed_wf = autoencoder.predict(X_test)
    plot_reconstructions(X_test, reconstructed_wf)


# ----------------- Main Function ----------------- #

def main():
    """
    Main function to parse arguments and run the specified task.
    """
    parser = argparse.ArgumentParser(
        description="Run supervised or unsupervised learning tasks for the SYK_ML project."
    )
    parser.add_argument(
        "task",
        choices=["supervised", "unsupervised"],
        help="The learning task to run.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=8,
        help="The number of Majorana fermions.",
    )
    args = parser.parse_args()

    if args.task == "supervised":
        run_supervised_task(args.N)
    elif args.task == "unsupervised":
        run_unsupervised_task(args.N)


if __name__ == "__main__":
    main()