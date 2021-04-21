# -*- coding: utf-8 -*-
"""
Mini project 2

Dennis Brown, COMP6636, 23 APR 2021
"""

import numpy as np
import random
import sys


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
        sys.stdout.flush()

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

        # Calculate hidden layer error (remove bias from hy)
        H_error = H_output * (1 - H_output) * np.dot(y_error, hy.T[:, 1:])

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
    xh, hy = train_neural_network(train_features, binary_train_classes,
                                  100, 1.0, 1000)

    # Test
    binary_pred_classes = test_neural_network(test_features, xh, hy)
    binary_pred_classes = 1.0 * (binary_pred_classes > 0.5)
    pred_classes = convert_mnist_classes_to_integer(binary_pred_classes)
    # print('Testing error:')
    # print(test_classes - pred_classes)
    correct = 0
    cm = np.zeros((10, 10))
    for i in range(test_classes.shape[0]):
        if (pred_classes[i][0] == test_classes[i][0]): correct += 1
        cm[int(pred_classes[i][0])][int(test_classes[i][0])] += 1
    print('Correct:', correct, '/', test_classes.shape[0])
    print(cm)

    return correct, cm


############################################
#
# Support Vector Machine functions
#
############################################

def train_svm(X, y, lam, limit):
    """
    Support Vector Machine. Given a sample matrix X,
    a vector Y of classifications, a regularization parameter lam,
    and a step limit, train and return a weight vector that
    can be used to classify the given data.
    """
    # Convert (1, 0) to (1, -1)
    y = y * 2 - 1

    # Initialize the weight vector
    w = np.zeros(X.shape[1])

    # Pegasos algorithm
    # Repeat the main loop until we reach the iteration limit
    t = 1
    while(t <= limit):
        i = random.randint(0, X.shape[0] - 1)
        eta = 1.0 / (lam * t)
        y_hat = y[i][0] * np.matmul(w, X[i])
        if (y_hat < 1.0):
            w = ((1 - (eta * lam)) * w) + (eta * y[i][0] * X[i])
        else:
            w = ((1 - (eta * lam)) * w)
        if (np.linalg.norm(w) > 0.0):
            w = min(1.0, ((1.0 / np.sqrt(lam)) / (np.linalg.norm(w)))) * w
        t += 1

    return w


def test_svm(X, w):
    """
    Support Vector Machine. Given a sample matrix X
    and a weight vector, predict the classes of X.
    """
    # Calculate predictions
    y_hat = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        y_hat[i][0] = np.matmul(w, X[i])

    # Convert to (1, -1)
    y_hat = np.sign(y_hat)

    # Convert (1, -1) to (1, 0)
    y_hat = (y_hat + 1) / 2

    return y_hat


def mnist_svm(train_classes, test_classes, train_features, test_features):
    """
    Given MNIST features and classes split into training and testing data,
    train and evaluate Support Vector Machine.
    """
    # Convert classes to four binary y vectors
    binary_train_classes = convert_mnist_classes_to_binary(train_classes)
    y1 = binary_train_classes[:,[0]]
    y2 = binary_train_classes[:,[1]]
    y3 = binary_train_classes[:,[2]]
    y4 = binary_train_classes[:,[3]]

    # Train on the four y vectors
    limit = 100 * train_features.shape[0]
    lam = 0.00001
    w1 = train_svm(train_features, y1, lam, limit)
    w2 = train_svm(train_features, y2, lam, limit)
    w3 = train_svm(train_features, y3, lam, limit)
    w4 = train_svm(train_features, y4, lam, limit)

    # Get binary predictions from the four perceptrons
    y_hat1 = test_svm(test_features, w1)
    y_hat2 = test_svm(test_features, w2)
    y_hat3 = test_svm(test_features, w3)
    y_hat4 = test_svm(test_features, w4)

    # Convert binary predictions back to decimal
    binary_pred_classes = np.hstack((y_hat1, y_hat2, y_hat3, y_hat4))
    pred_classes = convert_mnist_classes_to_integer(binary_pred_classes)

    # Calculate number correct
    correct = 0
    cm = np.zeros((10, 10))
    for i in range(test_classes.shape[0]):
        if (pred_classes[i][0] == test_classes[i][0]): correct += 1
        cm[int(pred_classes[i][0])][int(test_classes[i][0])] += 1
    print('Correct:', correct, '/', test_classes.shape[0])
    print(cm)

    return correct, cm


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


def gram(X):
    """
    Calculate Gram Matrix given X
    """
    G = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            G[i][j] = poly_kernel(X[i], X[j], 0.0, 1.0, 2.0)

    return G


def train_perceptron_kernel(G, y, beta, step_limit):
    """
    Perceptron with a kernel. Given a Gram matrix G,
    a vector Y of classifications, a learning rate (beta),
    and a step limit, train and return a weight vector that
    can be used to classify the given data.
    """
    # Convert (1, 0) to (1, -1)
    y = y * 2 - 1

    # Initialize the alpha vector
    a = np.zeros(G.shape[0])

    # Initialize y_hat
    y_hat = np.zeros((G.shape[0], 1))

    # Repeat the main loop until we have convergence or reach the
    # iteration limit
    steps = 0
    converged = False
    while(not(converged) and (steps < step_limit)):
        converged = True # assume converged until we determine otherwise

        # For each sample in X, calculate alpha's classification error
        # and update alpha.
        for i in range(G.shape[0]):

            # Find current prediction based on kernel
            y_hat[i][0] = np.sign(np.matmul(G[i,:], a))

            # If error on this element is > a very small value (is not
            # effectively 0), we need to update alpha, and have not converged.
            error = y[i][0] - y_hat[i][0]
            if (abs(error) > 0.000001):
                a[i] += beta * y[i][0]
                converged = False
        steps += 1

    # print('Final alpha = ', a, 'in', steps, 'steps; converged?', converged)

    return a


def test_perceptron_kernel(Xtrain, Xtest, a):
    """
    Perceptron with a kernel. Given a sample matrices Xtrain and Xtest,
    and vector a, return predicted classes.
    """
    y_hat = np.zeros((Xtest.shape[0], 1))

    for i in range(Xtest.shape[0]):
        for j in range(a.shape[0]):
            y_hat[i][0] += a[j] * poly_kernel(Xtrain[j], Xtest[i], 0.0, 1.0, 2.0)

    # Convert to (1, -1)
    y_hat = np.sign(y_hat)

    # Convert (1, -1) to (1, 0)
    y_hat = (y_hat + 1) / 2

    return y_hat


def mnist_perceptron_kernel(train_classes, test_classes, train_features,
                            test_features):
    """
    Given MNIST features and classes split into training and testing data,
    train and evaluate Kernel Perceptron.
    """
    # Convert classes to four binary y vectors
    binary_train_classes = convert_mnist_classes_to_binary(train_classes)
    y1 = binary_train_classes[:,[0]]
    y2 = binary_train_classes[:,[1]]
    y3 = binary_train_classes[:,[2]]
    y4 = binary_train_classes[:,[3]]

    # Train on the four y vectors
    limit = 5000
    beta = 1
    G = gram(train_features)
    a1 = train_perceptron_kernel(G, y1, beta, limit)
    a2 = train_perceptron_kernel(G, y2, beta, limit)
    a3 = train_perceptron_kernel(G, y3, beta, limit)
    a4 = train_perceptron_kernel(G, y4, beta, limit)

    # Get binary predictions from the four perceptrons
    y_hat1 = test_perceptron_kernel(train_features, test_features, a1)
    y_hat2 = test_perceptron_kernel(train_features, test_features, a2)
    y_hat3 = test_perceptron_kernel(train_features, test_features, a3)
    y_hat4 = test_perceptron_kernel(train_features, test_features, a4)

    # Convert binary predictions back to decimal
    binary_pred_classes = np.hstack((y_hat1, y_hat2, y_hat3, y_hat4))
    pred_classes = convert_mnist_classes_to_integer(binary_pred_classes)

    # Calculate number correct
    correct = 0
    cm = np.zeros((10, 10))
    for i in range(test_classes.shape[0]):
        if (pred_classes[i][0] == test_classes[i][0]): correct += 1
        cm[int(pred_classes[i][0])][int(test_classes[i][0])] += 1
    print('Correct:', correct, '/', test_classes.shape[0])
    print(cm)

    return correct, cm


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

    # # Test decimal-binary-decimal conversion
    # binary_train_classes = convert_mnist_classes_to_binary(train_classes)
    # decimal_train_classes = convert_mnist_classes_to_integer(binary_train_classes)
    # print(train_classes - decimal_train_classes)

    # Execute Neural Network testing
    # print('\nNeural Network')
    # acc_nn, confusion_nn = mnist_neural_network(train_classes, test_classes,
    #                                             train_features, test_features)
    # np.savetxt('./data/confusion_nn.csv', confusion_nn, delimiter=',', fmt='%10.0f')

    # Execute SVM testing
    print('\nSupport Vector Machine')
    acc_svm, confusion_svm = mnist_svm(train_classes, test_classes,
                                        train_features, test_features)
    np.savetxt('./data/confusion_svm.csv', confusion_svm, delimiter=',', fmt='%10.0f')

    # Execute Kernel Perceptron testing
    # print('\nKernel Perceptron')
    # acc_kp, confusion_kp = mnist_perceptron_kernel(train_classes, test_classes,
    #                                                train_features, test_features)
    # np.savetxt('./data/confusion_kp.csv', confusion_kp, delimiter=',', fmt='%10.0f')


if __name__ == '__main__':
    main()

