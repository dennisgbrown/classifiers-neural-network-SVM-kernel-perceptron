# -*- coding: utf-8 -*-
"""
Mini project 2

Dennis Brown, COMP6636, 23 APR 2021
"""

import numpy as np
import copy
import matplotlib.pyplot as plt


############################################
#
# MNIST data set functions
#
############################################

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


def convert_mnist_classes_to_binary(classes):
    """
    Given a list of integer MNIST classes, return an array where each class is
    converted to binary. e.g., 5 -> [0. 1. 0. 1.]
    """
    binary_classes = np.zeros((classes.shape[0], 4))
    for i in range(classes.shape[0]):
        boolver = bin(int(classes[i][0]))[2:].zfill(4)
        for bit in range(len(boolver)): binary_classes[i][bit] = float(boolver[bit])
        # print(classes[i][0], binary_classes[i])
    return binary_classes


def convert_mnist_classes_to_integer(binary_classes):
    """
    Given a list of binary MNIST classes, return an array where each class is
    converted to integer. e.g., [0. 1. 0. 1.] -> 5
    """
    classes = np.zeros((binary_classes.shape[0], 1))
    for i in range(binary_classes.shape[0]):
        bins = binary_classes[i]
        # Not very elegant, but it works -- convert to integer and cap at 9
        classes[i][0] = min((bins[0] * 8) + (bins[1] * 4) + (bins[2] * 2) + bins[3], 9)
        # print(classes[i][0], binary_classes[i])
    return classes


############################################
#
# Perceptron Kernel functions
#
############################################


def poly_kernel(x, z, a, b, d):
    """
    Calculate polynomial kernel for samples x and z.
    a, b, and d are hyperparameters.
    """
    return (a + (b * (np.matmul(x.T, z))) ** d)


def train_perceptron_kernel(X, y, beta, step_limit):
    """
    Perceptron with a kernel. Given a 2-D set of data X (samples are rows,
    columns features), a vector Y of classifications, a learning rate (beta),
    and a step limit, train and return a weight vector that
    can be used to classify the given data.
    """

    # Initialize the alpha vector
    a = np.zeros(X.shape[0])

    # Initialize y_hat
    y_hat = np.zeros((X.shape[0], 1))

    # Repeat the main loop until we have convergence or reach the
    # iteration limit
    steps = 0
    converged = False
    while(not(converged) and (steps < step_limit)):
        print(str(steps) + ' ', end = '')

        converged = True # assume converged until we determine otherwise

        # For each sample in X, calculate alpha's classification error
        # and update alpha.
        for i in range(len(X)):

            # Find current prediction based on kernel
            predict_val = 0
            for j in range(len(X)):
                predict_val += a[i] * poly_kernel(X[j], X[i], 0.0, 1.0, 1.0)
            y_hat[i][0] = 1 if (predict_val > 0) else -1

            # If error on this element is > a very small value (is not
            # effectively 0), we need to update alpha, and have not converged.
            error = y[i][0] - y_hat[i][0]
            if (abs(error) > 0.000001):
                a[i] += beta * y[i][0]
                converged = False
        steps += 1

    print()
    # print('Final alpha = ', a, 'in', steps, 'steps; converged?', converged)

    return a


def test_perceptron_kernel(X, a):
    y_hat = np.zeros((X.shape[0], 1))
    for i in range(len(X)):
        predict_val = 0
        for j in range(len(X)):
            predict_val += a[i] * poly_kernel(X[j], X[i], 0.0, 1.0, 1.0)
        y_hat[i][0] = 1 if (predict_val > 0) else -1
    # print('Y   :', y)
    # print('Y^  :', y_hat)
    # print('Diff:', y - y_hat)
    return y_hat


def mnist_perceptron_kernel(train_classes, test_classes, train_features,
                            test_features):

    binary_train_classes = convert_mnist_classes_to_binary(train_classes)

    y1 = binary_train_classes[:,[0]]
    y2 = binary_train_classes[:,[1]]
    y3 = binary_train_classes[:,[2]]
    y4 = binary_train_classes[:,[3]]

    limit = 3
    beta = .05
    a1 = train_perceptron_kernel(train_features, y1, beta, limit)
    a2 = train_perceptron_kernel(train_features, y2, beta, limit)
    a3 = train_perceptron_kernel(train_features, y3, beta, limit)
    a4 = train_perceptron_kernel(train_features, y4, beta, limit)

    y_hat1 = test_perceptron_kernel(train_features, a1)
    y_hat2 = test_perceptron_kernel(train_features, a2)
    y_hat3 = test_perceptron_kernel(train_features, a3)
    y_hat4 = test_perceptron_kernel(train_features, a4)

    binary_pred_classes = np.hstack((y_hat1, y_hat2, y_hat3, y_hat4))
    binary_pred_classes = 1.0 * (binary_pred_classes > 0.0)

    # print('YHAT', binary_pred_classes)

    pred_classes = convert_mnist_classes_to_integer(binary_pred_classes)
    correct = 0
    for i in range(test_classes.shape[0]):
        # print(test_classes[i][0], pred_classes[i][0])
        if (test_classes[i][0] == pred_classes[i][0]): correct += 1
    print('Correct:', correct, '/', test_classes.shape[0])

    """
    For 1000 samples:
    3 steps, beta .05: 26/300
    3 steps, beta .5: 26/300
    10 steps, beta .05: 26/300

    For 10000 samples:
    3 steps, beta .05: 302/3000
    """





############################################
#
# Neural Network functions
#
############################################

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
        print(str(epoch) + ' ', end = '')

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

    print()

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


def mnist_neural_network(train_classes, test_classes, train_features,
                         test_features):
    """
    Given MNIST features and classes split into training and testing data,
    train and evaluate Neural Network.
    """
    # Convert classifications to binary
    binary_train_classes = convert_mnist_classes_to_binary(train_classes)

    # Train
    xh, hy = train_neural_network(train_features, binary_train_classes, 100, 1.0, 100)


    # print(xh)
    # print(hy)

    # Test
    binary_pred_classes = test_neural_network(test_features, xh, hy)
    binary_pred_classes = 1.0 * (binary_pred_classes > 0.5)
    pred_classes = convert_mnist_classes_to_integer(binary_pred_classes)
    # print('Testing error:')
    # print(test_classes - pred_classes)
    correct = 0
    for i in range(test_classes.shape[0]):
        if (test_classes[i][0] == pred_classes[i][0]): correct += 1
    print('Correct:', correct, '/', test_classes.shape[0])

    """
    For 1000 samples:
    10 units, 10 epochs, LR 0.5: 33/300
    10 units, 1000 epochs, LR 0.5: 179/300
    10 units, 10000 epochs, LR 0.5: 167/300 and super slow
    20 units, 1000 epochs, LR 0.5: 191/300
    20 units, 10000 epochs, LR 0.5: 197/300 and super slow
    30 units, 1000 epochs, LR 0.5: 202/300
    100 units, 10 epochs, LR 0.5: 41/300 ?
    100 units, 1000 epochs, LR 0.5: 217/300
    100 units, 1000 epochs, LR 0.05: 155/300
    100 units, 1000 epochs, LR 5.0: 222/300

    For 10000 samples:
    10 units, 1000 epochs, LR 0.5: 2066/3000
    10 units, 1000 epochs, LR 1.0: 2229/3000
    100 units, 100 epochs, LR 1.0: /3000
    """


############################################
#
# Run it all
#
############################################

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

    # Execute Neural Network testing
    #mnist_neural_network(train_classes, test_classes, train_features, test_features)

    # Kernel Perceptron
    mnist_perceptron_kernel(train_classes, test_classes, train_features, test_features)

    # SVM



if __name__ == '__main__':
    main()

