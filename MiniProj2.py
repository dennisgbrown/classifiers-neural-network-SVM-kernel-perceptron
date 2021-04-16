# -*- coding: utf-8 -*-
"""
Mini project 2

Dennis Brown, COMP6636, 23 APR 2021
"""

import numpy as np
import copy
import matplotlib.pyplot as plt


def libsvm_scale_import(filename, limit = 0):
    """
    Read data from a libsvm .scale file. Set 'limit' to limit the import
    to a certain number of samples.
    """
    datafile = open(filename, 'r')

    # First pass: get dimensions of data
    num_samples = 0
    max_feature_id = 0
    for line in datafile:
        num_samples += 1
        tokens = line.split()
        for feature in tokens[1:]:
            feature_id = int(feature.split(':')[0])
            max_feature_id = max(feature_id, max_feature_id)

    # Second pass: read data into array
    # If the limit is set, import up to the limit, otherwise import all
    import_samples = min(num_samples, limit) if limit else num_samples
    features = np.zeros((import_samples, max_feature_id))
    classes = np.zeros((import_samples, 1))
    curr_sample = 0
    # Read the samples
    datafile.seek(0)
    for line in datafile:
        # Stop at the limit if it's set
        if (limit and (curr_sample >= limit)): break
        tokens = line.split()
        # Read the classification
        classes[curr_sample][0] = float(tokens[0])
        # Read the features
        for feature in tokens[1:]:
            feature_id = int(feature.split(':')[0])
            feature_val = float(feature.split(':')[1])
            features[curr_sample][feature_id - 1] = feature_val
        curr_sample += 1
    datafile.close()

    print('LOADED:', filename, ':', classes.shape, features.shape)

    return classes, features


def convert_classes_to_binary(classes):
    """
    Given a list of integer classes, return an array where each class is
    converted to binary. e.g., 5.0 -> [0. 1. 0. 1.]
    """
    binary_classes = np.zeros((classes.shape[0], 4))
    for i in range(classes.shape[0]):
        boolver = bin(int(classes[i][0]))[2:].zfill(4)
        for bit in range(len(boolver)): binary_classes[i][bit] = float(boolver[bit])
        # print(classes[i][0], binary_classes[i])
    return binary_classes


# TODO: Do we need this???
def convert_classes_to_integer(binary_classes):
    classes = np.zeros((binary_classes.shape[0], 1))
    for i in range(binary_classes.shape[0]):
        pass
        # print(classes[i][0], binary_classes[i])
    return classes



def sigmoid(x):
    """
    Sigmoid Function
    """
    return 1 / (1 + np.exp(-x))


def train_neural_network(X, y, H_size, learning_rate, epochs):
    """
    2-Layer Neural Network: Given a 2-D set of data X (samples are rows,
    columns features), a vector y of classifications, a number of
    hidden-layer neurons, a learning rate, and number of epochs,
    train a 2-layer neural network. Return weight matrices.
    """

    # Randomly initialize the weights for the input -> hidden layer
    xh = (np.random.random((X.shape[1] + 1, H_size))) * 2 - 1

    # Randomly initialize the weights for the hidden layer -> output
    hy = (np.random.random((H_size + 1, y.shape[1]))) * 2 - 1

    for epoch in range(epochs):

        # --------------------
        # Forward Propagation
        # --------------------

        # Add bias terms to X
        X_bias = np.hstack((X, np.ones([X.shape[0], 1])))

        # Calculate hidden layer outputs
        H_output = sigmoid(np.dot(X_bias, xh))

        # Add bias terms to H_output
        H_output_bias = np.hstack((H_output, (np.ones([H_output.shape[0], 1]))))

        y_hat = sigmoid(np.dot(H_output_bias, hy))

        # --------------------
        # Backward Propagation
        # --------------------

        # Find error
        y_error = y_hat - y

        # Calculate hidden layer error
        H_error = H_output * (1 - H_output) * np.dot(y_error, hy.T[:, 1:]) # remove bias from hy

        # Calculate partial derivatives
        H_pd = X_bias[:, :, np.newaxis] * H_error[: , np.newaxis, :]
        y_pd = H_output_bias[:, :, np.newaxis] * y_error[:, np.newaxis, :]

        # Calculate total gradients for hidden and output layers
        # (find average of each column)
        H_gradient = np.average(H_pd, axis = 0)
        y_gradient = np.average(y_pd, axis = 0)

        # Update weights using learning rate and gradients
        xh -= (learning_rate * H_gradient)
        hy -= (learning_rate * y_gradient)

    # Return weight matrices when finished
    return xh, hy


def test_neural_network(X, xh, hy):
    """
    2-layer Neural Network: Given a 2-D set of data X (samples are rows,
    columns features) and weight matrices for the 2-layer network,
    return the predicted output.
    """
    X_bias = np.hstack((X, np.ones([X.shape[0], 1])))
    H_output = sigmoid(np.dot(X_bias, xh))
    H_output_bias = np.hstack((H_output, (np.ones([H_output.shape[0], 1]))))
    y_hat = sigmoid(np.dot(H_output_bias, hy))
    return y_hat


def main():
    # Load data
    classes, features = libsvm_scale_import('data/mnist.scale', limit = 1000)
    split = int(len(classes) * 0.70)
    train_classes = classes[:split]
    test_classes = classes[split:]
    train_features = features[:split]
    test_features = features[split:]
    print('training data =', train_features.shape, train_classes.shape)
    print('test_data =', test_features.shape, test_classes.shape)

    # 2-Layer Neural network

    # Convert classifications to binary
    binary_train_classes = convert_classes_to_binary(train_classes)

    # Train
    xh, hy = train_neural_network(train_features, binary_train_classes, 10, 0.5, 10000)

    # print(xh)
    # print(hy)

    # Test
    binary_test_classes = convert_classes_to_binary(test_classes)
    predictions = test_neural_network(test_features, xh, hy)
    predictions = 1.0 * (predictions > 0.5)
    # print(predictions)
    # print('Testing error:')
    # print(binary_test_classes - predictions)
    correct = 0
    for i in range(binary_test_classes.shape[0]):
        if (np.array_equal(binary_test_classes[i], predictions[i])): correct += 1
    print('Correct:', correct, '/', binary_test_classes.shape[0])


if __name__ == '__main__':
    main()

