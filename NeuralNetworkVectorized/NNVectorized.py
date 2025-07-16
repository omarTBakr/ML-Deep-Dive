"""
Neural Network Vectorized Implementation for Multiclass Classification
Made by GPT with the assistance of the author.

This code is for educational purposes, focusing on:
1. Deep understanding of neural networks
2. Implementation skill

########################################################################################
In this file, we implement a neural network in a vectorized manner.
Naming convention:
    net = W * input to the layer
    out = activation(net)
########################################################################################
"""

import numpy as np
from loaddata import load_mnist_data


# --- Vectorized activation functions and derivatives ---


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    """Derivative of sigmoid activation."""
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)


def tanh_der(x):
    """Derivative of tanh activation."""
    return 1 - np.tanh(x) ** 2


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_der(x):
    """Derivative of ReLU activation."""
    return (x > 0).astype(float)


def softmax(x):
    """
    Softmax activation function for output layer (multiclass classification).
    Args:
        x: Input array of shape (n_classes, n_samples) or (n_classes,)
    Returns:
        Softmax probabilities of same shape as input.
    """
    x = x - np.max(x, axis=0, keepdims=True)  # For numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def identity(x):
    """Identity activation (no change)."""
    return x


def identity_der(x):
    """Derivative of identity activation (always 1)."""
    return np.ones_like(x)


# Dictionary mapping activation names to functions and their derivatives
activation_functions = {
    "sigmoid": (sigmoid, sigmoid_der),
    "tanh": (tanh, tanh_der),
    "relu": (relu, relu_der),
    "identity": (identity, identity_der),
}


class NeuralLayer:
    """
    Represents a single fully-connected (dense) neural network layer.
    Handles forward and backward propagation for this layer.
    """

    def __init__(self, in_dim, out_dim, activation, weights=None):
        """
        Args:
            in_dim (int): Number of input features.
            out_dim (int): Number of output neurons.
            activation (str): Activation function name.
            weights (np.ndarray, optional): Custom weight matrix.
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_name = activation
        self.activation, self.activation_der = activation_functions[activation]
        # Xavier/He initialization if weights not provided
        if weights is not None:
            self.W = weights
        else:
            limit = np.sqrt(6 / (in_dim + out_dim))
            self.W = np.random.uniform(-limit, limit, (out_dim, in_dim))
        self.b = np.zeros((out_dim, 1))

    def forward(self, X):
        """
        Forward pass through the layer.
        Args:
            X (np.ndarray): Input of shape (in_dim, n_samples)
        Returns:
            np.ndarray: Output after activation (out_dim, n_samples)
        """
        self.X = X
        self.Z = self.W @ X + self.b  # Linear transformation
        self.A = self.activation(self.Z)  # Apply activation
        return self.A

    def backward(self, dA):
        """
        Backward pass: computes gradients for weights and bias.
        Args:
            dA (np.ndarray): Gradient from next layer (out_dim, n_samples)
        Returns:
            np.ndarray: Gradient to pass to previous layer (in_dim, n_samples)
        """
        m = self.X.shape[1]
        dZ = dA * self.activation_der(self.Z)
        self.dW = (1 / m) * dZ @ self.X.T
        self.db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dX = self.W.T @ dZ
        return dX

    def update(self, lr):
        """
        Update weights and biases using computed gradients.
        Args:
            lr (float): Learning rate
        """
        self.W -= lr * self.dW
        self.b -= lr * self.db


class NeuralNetwork:
    """
    Represents a simple feedforward neural network for multiclass classification.
    Handles forward, backward, and training steps.
    """

    def __init__(self, layer_sizes, activations, lr=0.01, weights=None):
        """
        Args:
            layer_sizes (list): List of layer sizes [input_dim, hidden1, ..., output_dim]
            activations (list): List of activation names for each layer except input
            lr (float): Learning rate
            weights (list, optional): List of custom weight matrices
        """
        assert len(layer_sizes) - 1 == len(activations)
        self.lr = lr
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            w = weights[i] if weights is not None else None
            self.layers.append(
                NeuralLayer(
                    layer_sizes[i], layer_sizes[i + 1], activations[i], weights=w
                )
            )

    def forward(self, X):
        """
        Forward pass through the network.
        Args:
            X (np.ndarray): Input data (input_dim, n_samples)
        Returns:
            np.ndarray: Output logits (output_dim, n_samples)
        """
        for i, layer in enumerate(self.layers):
            X = layer.forward(X)
        return X

    def compute_loss(self, Y_hat, Y):
        """
        Compute cross-entropy loss for multiclass classification.
        Args:
            Y_hat (np.ndarray): Predicted probabilities (output_dim, n_samples)
            Y (np.ndarray): True one-hot labels (output_dim, n_samples)
        Returns:
            float: Cross-entropy loss
        """
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(Y_hat + 1e-8)) / m
        return loss

    def backward(self, Y_hat, Y):
        """
        Backward pass through the network (computes gradients).
        Args:
            Y_hat (np.ndarray): Predicted probabilities (output_dim, n_samples)
            Y (np.ndarray): True one-hot labels (output_dim, n_samples)
        """
        m = Y.shape[1]
        dA = Y_hat - Y  # for softmax + cross-entropy
        for i in reversed(range(len(self.layers))):
            dA = self.layers[i].backward(dA)

    def update(self):
        """
        Update all network weights and biases using gradients.
        """
        for layer in self.layers:
            layer.update(self.lr)

    def train_step(self, X, Y):
        """
        Perform one training step (forward, backward, update).
        Args:
            X (np.ndarray): Input data (input_dim, n_samples)
            Y (np.ndarray): True one-hot labels (output_dim, n_samples)
        Returns:
            float: Loss value for this step
        """
        Y_hat = self.forward(X)
        loss = self.compute_loss(Y_hat, Y)
        self.backward(Y_hat, Y)
        self.update()
        return loss


# --- Utility functions for data processing ---
def dataset_to_numpy(dataset):
    """
    Convert a torchvision MNIST dataset to numpy arrays.
    Args:
        dataset: torchvision.datasets.MNIST object
    Returns:
        X: np.ndarray of shape (784, n_samples), normalized images
        Y: np.ndarray of shape (n_samples,), integer labels
    """
    n_samples = len(dataset)
    X = np.zeros((28 * 28, n_samples), dtype=np.float32)
    Y = np.zeros((n_samples,), dtype=np.int64)
    for i in range(n_samples):
        img, label = dataset[i]
        X[:, i] = img.view(-1).numpy().astype(np.float32)
        Y[i] = label
    return X, Y


def one_hot(y, num_classes=10):
    """
    One-hot encode integer labels.
    Args:
        y: np.ndarray of shape (n_samples,)
        num_classes: int, number of classes
    Returns:
        Y: np.ndarray of shape (num_classes, n_samples)
    """
    Y = np.zeros((num_classes, y.size), dtype=np.float32)
    Y[y, np.arange(y.size)] = 1.0
    return Y


if __name__ == "__main__":
    """
    Main script for training and evaluating a simple neural network on MNIST.
    Steps:
    1. Load MNIST data using torchvision
    2. Convert to numpy arrays and preprocess (flatten, normalize, one-hot)
    3. Build and train a neural network
    4. Print training and validation loss/accuracy every 10 epochs
    """
    # Load MNIST data (returns train_dataset, test_dataset)
    train_dataset, test_dataset = load_mnist_data()

    # Convert datasets to numpy arrays
    X_train, y_train = dataset_to_numpy(train_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)

    # Normalize images to [0, 1]
    X_train /= 255.0 if X_train.max() > 1 else 1.0
    X_test /= 255.0 if X_test.max() > 1 else 1.0

    # One-hot encode labels
    Y_train = one_hot(y_train, 10)
    Y_test = one_hot(y_test, 10)

    # Define network architecture
    input_dim = X_train.shape[0]
    output_dim = Y_train.shape[0]
    hidden_units = 64  # You can change this for more/less capacity

    # Create the neural network
    net = NeuralNetwork(
        [input_dim, hidden_units, output_dim],
        activations=[
            "relu",  # Hidden layer activation
            "identity",  # Output layer: identity, softmax applied outside
        ],
        lr=0.1,
    )

    epochs = 100
    for epoch in range(epochs):
        # Forward pass on training data
        logits = net.forward(X_train)
        Y_hat = softmax(logits)
        loss = net.compute_loss(Y_hat, Y_train)
        # Backward pass and update
        net.backward(Y_hat, Y_train)
        net.update()

        if (epoch + 1) % 10 == 0:
            # Validation on test data
            logits_val = net.forward(X_test)
            Y_hat_val = softmax(logits_val)
            val_loss = net.compute_loss(Y_hat_val, Y_test)
            preds = np.argmax(Y_hat_val, axis=0)
            labels = np.argmax(Y_test, axis=0)
            acc = np.mean(preds == labels)
            print(
                f"Epoch {epoch + 1}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {acc:.4f}"
            )
