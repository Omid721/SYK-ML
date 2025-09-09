"""
This script implements a supervised neural network to classify SYK ground states.
The model is trained on a dataset of SYK ground states and Haar random states.
The performance of the model is evaluated by its accuracy on a test set.
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
    # Shuffle the data
    data = shuffle(data, random_state=0)

    # Define the target and features
    target = "is_SYK"
    features = data.columns.drop(target)

    # Split the data into training and testing sets
    X = data[features]
    y = data[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, features


def build_nn_model(input_shape):
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
            tf.keras.layers.Dense(
                1, activation="sigmoid"
            ),  # Sigmoid for binary classification
        ]
    )
    return model


def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    """Train the neural network model.

    Args:
        model (tf.keras.Sequential): The neural network model.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        epochs (int): The number of epochs to train for.
        batch_size (int): The batch size for training.
    """
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)


def evaluate_model(model, X_test, y_test):
    """Evaluate the neural network model.

    Args:
        model (tf.keras.Sequential): The neural network model.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing target.

    Returns:
        tuple: A tuple containing the loss and accuracy.
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return loss, accuracy


def plot_results(results):
    """Plot the loss and accuracy of the models.

    Args:
        results (dict): A dictionary containing the loss and accuracy for each N.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot Loss
    ax1.bar(results.keys(), [r["loss"] for r in results.values()], color="red")
    ax1.set_title("Loss")
    ax1.set_ylabel("Loss Value")

    # Plot Accuracy
    ax2.bar(
        results.keys(), [r["accuracy"] for r in results.values()], color="blue"
    )
    ax2.set_title("Accuracy")
    ax2.set_ylabel("Accuracy Value")

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the supervised learning experiment.
    """
    # Define the file paths for the datasets
    N_values = [8]
    file_paths = {f"N{N}": f"SYK_data_N{N}.csv" for N in N_values}

    # Dictionary to store the results
    results = {}

    # Iterate over the datasets
    for key, path in file_paths.items():
        print(f"Processing {key} data...")

        # Load and preprocess the data
        data = load_data(path)
        X_train, X_test, y_train, y_test, features = preprocess_data(data)

        # Build, train, and evaluate the model
        model = build_nn_model(len(features))
        train_model(model, X_train, y_train)
        loss, accuracy = evaluate_model(model, X_test, y_test)

        # Store the results
        results[key] = {"loss": loss, "accuracy": accuracy}
        print(f"Test Loss for {key}: {loss:.4f}")
        print(f"Test Accuracy for {key}: {accuracy:.4f}")

    # Plot the results
    plot_results(results)


if __name__ == "__main__":
    main()