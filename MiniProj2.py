# -*- coding: utf-8 -*-
"""
Mini project 2

Dennis Brown, COMP6636, 23 APR 2021
"""

import numpy as np
import random
import sys
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
                         test_features, h_size, learning_rate, num_epochs):
    """
    Given MNIST features and classes split into training and testing data,
    train and evaluate Neural Network.
    """
    # Convert classifications to binary
    binary_train_classes = convert_mnist_classes_to_binary(train_classes)

    # Train
    xh, hy = train_neural_network(train_features, binary_train_classes,
                                  h_size, learning_rate, num_epochs)

    # Test
    binary_pred_classes = test_neural_network(test_features, xh, hy)
    binary_pred_classes = 1.0 * (binary_pred_classes > 0.5)
    pred_classes = convert_mnist_classes_to_integer(binary_pred_classes)

    # Create label for this evaluation
    label = str(train_features.shape[0] + test_features.shape[0])
    label += '_' + str(h_size)
    label += '_' + str(learning_rate)
    label += '_' + str(num_epochs)

    # Calculate number correct
    correct = 0
    cm = np.zeros((10, 10))
    for i in range(test_classes.shape[0]):
        if (pred_classes[i][0] == test_classes[i][0]): correct += 1
        cm[int(pred_classes[i][0])][int(test_classes[i][0])] += 1
    print('Correct:', correct, '/', test_classes.shape[0], 'for', label)
    print(cm)

    np.savetxt('./data/confusion_nn_' + label + '.csv', cm, delimiter=',', fmt='%10.0f')

    return correct / test_classes.shape[0]


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


def mnist_svm(train_classes, test_classes, train_features, test_features,
              limit, lam):
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

    # Create label for this evaluation
    label = str(train_features.shape[0] + test_features.shape[0])
    label += '_' + str(limit)
    label += '_' + str(lam)

    # Calculate number correct
    correct = 0
    cm = np.zeros((10, 10))
    for i in range(test_classes.shape[0]):
        if (pred_classes[i][0] == test_classes[i][0]): correct += 1
        cm[int(pred_classes[i][0])][int(test_classes[i][0])] += 1
    print('Correct:', correct, '/', test_classes.shape[0], 'for', label)
    print(cm)

    np.savetxt('./data/confusion_svm_' + label + '.csv', cm, delimiter=',', fmt='%10.0f')

    return correct / test_classes.shape[0]


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


def gram(X, ka, kb, kd):
    """
    Calculate Gram Matrix given X and parameters for poly kernel
    """
    G = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            G[i][j] = poly_kernel(X[i], X[j], ka, kb, kd)

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

    return a


def test_perceptron_kernel(Xtrain, Xtest, a, ka, kb, kd):
    """
    Perceptron with a kernel. Given a sample matrices Xtrain and Xtest,
    and vector a, return predicted classes.
    """
    y_hat = np.zeros((Xtest.shape[0], 1))

    for i in range(Xtest.shape[0]):
        for j in range(a.shape[0]):
            y_hat[i][0] += a[j] * poly_kernel(Xtrain[j], Xtest[i], ka, kb, kd)

    # Convert to (1, -1)
    y_hat = np.sign(y_hat)

    # Convert (1, -1) to (1, 0)
    y_hat = (y_hat + 1) / 2

    return y_hat


def mnist_perceptron_kernel(train_classes, test_classes, train_features,
                            test_features, limit, beta, ka, kb, kd):
    """
    Given MNIST features and classes split into training and testing data,
    train and evaluate Kernel Perceptron. ka, kb, and kd are for poly kernel.
    """
    # Convert classes to four binary y vectors
    binary_train_classes = convert_mnist_classes_to_binary(train_classes)
    y1 = binary_train_classes[:,[0]]
    y2 = binary_train_classes[:,[1]]
    y3 = binary_train_classes[:,[2]]
    y4 = binary_train_classes[:,[3]]

    # Train on the four y vectors
    G = gram(train_features, ka, kb, kd)
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

    # Create label for this evaluation
    label = str(train_features.shape[0] + test_features.shape[0])
    label += '_' + str(limit)
    label += '_' + str(beta)
    label += '_' + str(ka)
    label += '_' + str(kb)
    label += '_' + str(kd)

    # Calculate number correct
    correct = 0
    cm = np.zeros((10, 10))
    for i in range(test_classes.shape[0]):
        if (pred_classes[i][0] == test_classes[i][0]): correct += 1
        cm[int(pred_classes[i][0])][int(test_classes[i][0])] += 1
    print('Correct:', correct, '/', test_classes.shape[0], 'for', label)
    print(cm)

    np.savetxt('./data/confusion_kp_' + label + '.csv', cm, delimiter=',', fmt='%10.0f')

    return correct / test_classes.shape[0]


############################################
#
# Run it all
#
############################################

def main():

    # Load data
    sample_limit = 1000
    classes, features = libsvm_scale_import('data/mnist.scale', limit = sample_limit)
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
    print('\nNeural Network')
    # nn_lrs = np.array([0.01, 0.1, 1.0, 10.0])
    # nn_lr_results = np.zeros(nn_lrs.shape)
    # for i in range(nn_lrs.shape[0]):
    #     nn_lr_results[i] = mnist_neural_network(train_classes, test_classes, train_features,
    #                                             test_features, 100, nn_lrs[i], 100)
    # plt.clf()
    # plt.plot(nn_lrs, nn_lr_results, marker='.')
    # plt.title('Neural Network: accuracy vs. learning rate for h=100, epochs=100')
    # plt.xscale('log')
    # plt.xlabel('learning rate')
    # plt.ylabel('accuracy')
    # plt.ylim(bottom = 0)
    # plt.grid(True)
    # plt.savefig('./plots/nn_accuracy_learning_rate.png', dpi = 600)

    # nn_hs = np.array([1, 10, 100, 1000])
    # nn_h_results = np.zeros(nn_hs.shape)
    # for i in range(nn_hs.shape[0]):
    #     nn_h_results[i] = mnist_neural_network(train_classes, test_classes, train_features,
    #                                            test_features, nn_hs[i], 1.0, 100)
    # plt.clf()
    # plt.plot(nn_hs, nn_h_results, marker='.')
    # plt.title('Neural Network: accuracy vs. hidden layer size for lr=1.0, epochs=100')
    # plt.xscale('log')
    # plt.xlabel('hidden layer size')
    # plt.ylabel('accuracy')
    # plt.ylim(bottom = 0)
    # plt.grid(True)
    # plt.savefig('./plots/nn_accuracy_hsize.png', dpi = 600)

    nn_epochs = np.array([10, 100, 1000, 10000])
    nn_epoch_results = np.zeros(nn_epochs.shape)
    for i in range(nn_epochs.shape[0]):
        nn_epoch_results[i] = mnist_neural_network(train_classes, test_classes, train_features,
                                                   test_features, 100, 1.0, nn_epochs[i])
    plt.clf()
    plt.plot(nn_epochs, nn_epoch_results, marker='.')
    plt.title('Neural Network: accuracy vs. epochs for h=100, lr=1.0')
    plt.xscale('log')
    plt.xlabel('number of epochs')
    plt.ylabel('accuracy')
    plt.ylim(bottom = 0)
    plt.grid(True)
    plt.savefig('./plots/nn_accuracy_epochs.png', dpi = 600)

    # # Execute SVM testing
    # print('\nSupport Vector Machine')
    # mnist_svm(train_classes, test_classes, train_features, test_features,
    #           100000, 0.00001)

    # # Execute Kernel Perceptron testing
    # print('\nKernel Perceptron')
    # mnist_perceptron_kernel(train_classes, test_classes, train_features,
    #                         test_features, 5000, 1, 0.0, 1.0, 2.0)


if __name__ == '__main__':
    main()

