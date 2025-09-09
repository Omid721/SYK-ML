# SYK_ML: Machine Learning for the Sachdev-Ye-Kitaev Model

This repository contains code for exploring the properties of the Sachdev-Ye-Kitaev (SYK) model's ground state using neural networks. The primary goal is to classify whether a given wave function originates from an SYK Hamiltonian and to analyze the model's ground state features.

## Introduction

The SYK model is a quantum mechanical model of interacting Majorana fermions. It has recently gained attention due to its connection to black holes and quantum chaos. This project investigates the complexity of the SYK model's ground state, which is known to be more intricate than a simple Gaussian state. We employ machine learning techniques, specifically neural networks, to analyze and classify these ground states.

## Features

-   **Supervised Learning:** A supervised neural network is used to classify wave functions as either SYK ground states or Haar random states. The model is trained on a dataset of SYK ground states and Haar random states and evaluated based on its classification accuracy.
-   **Unsupervised Learning:** An autoencoder neural network is used to learn the essential features of the SYK model's ground state. The autoencoder is trained on a dataset of SYK ground states and its performance is evaluated by its ability to reconstruct the original wave functions.
-   **Data Generation:** The repository includes scripts to generate training and testing data, including SYK ground states and Haar random states for various system sizes.

## Project Structure

The repository is organized as follows:

-   `data_genration.py`: A script to generate the training and testing data.
-   `Supervised_NN_SYK.py`: A script that implements the supervised neural network for classifying SYK ground states.
-   `Unsupervised_NN_SYK.py`: A script that implements the autoencoder neural network for learning the features of the SYK ground state.
-   `SYK_ML.py`: A script that combines the supervised and unsupervised learning approaches.
-   `README.md`: This file.

## Installation

To run the code in this repository, you will need to have the following libraries installed:

-   `numpy`
-   `scipy`
--  `pandas`
-   `tensorflow`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`

You can install these libraries using `pip`:

```bash
pip install numpy scipy pandas tensorflow scikit-learn matplotlib seaborn
```

## Usage

To run the code, you can use the following commands:

1.  **Generate Data:**
    ```bash
    python data_genration.py
    ```
2.  **Run Supervised Learning:**
    ```bash
    python Supervised_NN_SYK.py
    ```
3.  **Run Unsupervised Learning:**
    ```bash
    python Unsupervised_NN_SYK.py
    ```

## Contributing

Contributions to this project are welcome. If you would like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with a descriptive message.
4.  Push your changes to your forked repository.
5.  Create a pull request to the main repository.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.