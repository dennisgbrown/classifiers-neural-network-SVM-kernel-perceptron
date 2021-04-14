# -*- coding: utf-8 -*-
"""
Mini project 1

Dennis Brown, COMP6636, 03 MAR 2021
"""

import numpy as np
import copy
import matplotlib.pyplot as plt


def libsvm_scale_import(filename):
    """
    Read data from a libsvm .scale file
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
    data = np.zeros((num_samples, max_feature_id + 1))
    curr_sample = 0
    datafile.seek(0)
    for line in datafile:
        tokens = line.split()
        data[curr_sample][0] = float(tokens[0])
        for feature in tokens[1:]:
            feature_id = int(feature.split(':')[0])
            feature_val = float(feature.split(':')[1])
            data[curr_sample][feature_id] = feature_val
        curr_sample += 1
    datafile.close()

    print('LOADED:', filename, ':', data.shape)

    return data


def get_neighbors(data, test_sample, num_neighbors):
    """
    Given training data, a test sample, and a number of
    neighbors, return the closest neighbors.
    """
    # Calculate all distances from the training samples
    # to this test sample. Collect index, distance into a list.
    indices_and_distances = list()
    for i in range(len(data)):
        dist = np.linalg.norm(test_sample[1:] - (data[i])[1:]) # leave out classification at pos 0
        indices_and_distances.append([i, dist])

    # Sort list by distance
    indices_and_distances.sort(key=lambda _: _[1])

    # Make a list of requested number of closest neighbors from sorted
    # list of indices+distances
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(indices_and_distances[i][0])

    return neighbors


def classify_one_sample(data, test_sample, num_neighbors):
    """
    Given training data, a test sample, and a number of neighbors,
    predict which classification the test sample belongs to.
    """
    # Get closest neighbors
    neighbors = get_neighbors(data, test_sample, num_neighbors)

    # Create list of classifications of the neighbors
    classifications = list()
    for i in range(len(neighbors)):
        classifications.append(data[neighbors[i]][0]) # 0 = classification

    # Return the most common classification of the neighbors
    prediction = max(set(classifications), key = classifications.count)
    return prediction


def k_nearest_neighbors(data, test_samples, num_neighbors):
    """
    Given sample data (samples are rows, columns
    features, and samples have classifications in position 0),
    test data, and a number of neighbors, predict which classification
    each test sample belongs to.
    """
    classifications = list()
    for i in range(len(test_samples)):
        output = classify_one_sample(data, test_samples[i], num_neighbors)
        classifications.append(output)
        if ((i % 20) == 0):
            print('\rknn test sample', i, end='')
    print()

    return(classifications)


def check_knn_classifications(y, y_hat):
    """
    Given actual values y and classiciations y_hat,
    return the number of errors
    """
    errors = 0
    for i in range(len(y)):
        if (y[i] != y_hat[i]):
            errors += 1

    return errors


def train_perceptron(data, beta, step_limit):
    """
    Perceptron. Given a set of data (samples are rows, columns
    features, and samples have classifications in position 0),
    a step size (beta), and a step limit, train and return a
    weight vector that can be used to classify the given data.
    """

    # Initialize the weight vector including bias element
    w = np.zeros(len(data[0]))

    # Initialize y_hat
    y_hat = np.zeros(len(data))

    # Slice off y
    y = data[:,0]

    # Repeat the main loop until we have convergence or reach the
    # iteration limit
    steps = 0
    converged = False
    while(not(converged) and (steps < step_limit)):
        converged = True

        # For each sample in the data, calculate w's classification error
        # and update w.
        for i in range(len(data)):
            # Replace classification in sample[0] with a 1 to allow
            # for a biased weight vector
            biased_sample = np.copy(data[i])
            biased_sample[0] = 1

            # Get prediction and error, then update weight vector
            y_hat[i] = 1 if (np.matmul(w.T, biased_sample) > 0) else -1
            error = y[i] - y_hat[i]
            w += biased_sample * error * beta
            steps += 1

            # If error on this element is > a very small value, we have
            # not converged.
            if (abs(error) > 0.000001):
                converged = False

    print('Perceptron:' ,steps, 'steps; converged?', converged)

    return w


def multiclass_train_perceptron(data, beta, step_limit):
    """
    Perceptron. Given a set of data (samples are rows, columns
    features, and samples have classifications in position 0),
    a step size (beta), and a step limit, train and return a
    weight vector that can be used to classify the given data.

    This version works on data with multiple classes by one-vs-rest.
    """
    # Find unique classes
    classes = []
    for i in range(data.shape[0]):
        if (not(data[i][0] in classes)):
            classes.append(data[i][0])

    # For each classification, train perceptron on current class vs.
    # rest of the untrained classes.
    ws = []
    curr_data = copy.deepcopy(data)
    for curr_class in range(len(classes)):

        # Save original classification data
        orig_classes = copy.deepcopy(curr_data[:,0])

        # Reset classification data to 1 (for current class) or -1 for other
        for i in range(curr_data.shape[0]):
            if (curr_data[i][0] == classes[curr_class]):
                curr_data[i][0] = 1
            else:
                curr_data[i][0] = -1

        # Train and find weights
        ws.append(train_perceptron(curr_data, beta, step_limit))

        # Put original classifications back
        for i in range(curr_data.shape[0]):
            curr_data[i][0] = orig_classes[i]

    return ws


def test_perceptron(data, w):
    """
    Given test data and a weight vector w, return number of
    num_misclass when classifying the test data using the
    weights.
    """
    errors = 0

    # Initialize y_hat
    y_hat = np.zeros(len(data))

    # Slice off y
    y = data[:,0]

    # Determine how weights classify each test sample and count
    # num_misclass
    for i in range(len(data)):
        biased_sample = np.copy(data[i])
        biased_sample[0] = 1
        y_hat[i] = 1 if (np.matmul(w.T, biased_sample) > 0) else -1
        if (y[i] != y_hat[i]):
            errors += 1

    return errors


def multiclass_test_perceptron(data, ws):
    """
    Given test data and a weight vector w, return number of
    num_misclass when classifying the test data using the
    weights.

    This version works on data with multiple classes by One vs. All (OVA).
    """
    # Find unique classes
    classes = []
    for i in range(data.shape[0]):
        if (not(data[i][0] in classes)):
            classes.append(data[i][0])

    # For each classification, test perceptron on current class vs.
    # rest of the untested classes.
    errors = []
    curr_data = copy.deepcopy(data)
    for curr_class in range(len(classes)):

        # Save original classification data
        orig_classes = copy.deepcopy(curr_data[:,0])

        # Reset classification data to 1 (for current class) or -1 for other
        for i in range(curr_data.shape[0]):
            if (curr_data[i][0] == classes[curr_class]):
                curr_data[i][0] = 1
            else:
                curr_data[i][0] = -1

        # Train and find weights
        errors.append(test_perceptron(curr_data, ws[curr_class]))

        # Put original classifications back
        for i in range(curr_data.shape[0]):
            curr_data[i][0] = orig_classes[i]

    return errors


def iris_knn():
    """
    Run kNN on the iris dataset for the various numbers of neighbors.
    """
    print("----------\niris kNN")

    # Load data
    data = libsvm_scale_import('data/iris.scale')

    # Shuffle the data because we want to split it into train & test,
    # and it is pre-sorted (we would test against classes we didn't
    # see in training)
    np.random.seed(1) # ensure consistent shuffling
    np.random.shuffle(data)

    # Split up data into training and test data based on split value
    split = 50
    train_data = data[:split]
    test_data = data[split:]

    # Test multiple values of k
    test_ks = np.arange(1, split)
    error_rates = np.zeros(test_ks.shape[0])
    for i in range(len(test_ks)):
        # Classify the test data
        print('Classify with k =', test_ks[i])
        classifications = k_nearest_neighbors(train_data, test_data,
                                              test_ks[i])
        # Check accuracy
        errors = check_knn_classifications(test_data[:,0], classifications)
        error_rates[i] = errors / test_data.shape[0]
        print(errors, 'errors in', test_data.shape[0], 'samples')

    print('ks:', test_ks)
    print('error rates:', error_rates)
    plt.clf()
    plt.plot(test_ks, error_rates, marker='.')
    plt.title('Iris kNN: error rate vs. k')
    plt.xlabel('k')
    plt.ylabel('error rate')
    plt.xlim(left = 0)
    plt.ylim(bottom = 0)
    plt.grid(True)
    plt.savefig('iris_knn.png', dpi = 600)


def iris_perceptron():
    """
    Run Perceptron on the iris dataset in various ways.
    """
    print("----------\niris Perceptron")

    # Load data
    data = libsvm_scale_import('data/iris.scale')

    # Shuffle the data because we want to split it into train & test,
    # and it is pre-sorted (we would test against classes we didn't
    # see in training)
    np.random.seed(1) # ensure consistent shuffling
    np.random.shuffle(data)

    # Split up data into training and test data based on split value
    split = 50
    train_data = data[:split]
    test_data = data[split:]

    # Perform multi-class training and test and collect
    # a weight vector and number of errors for each class
    ws = multiclass_train_perceptron(train_data, 0.1, 100000)
    errors = multiclass_test_perceptron(test_data, ws)

    # Report errors
    print(errors, 'errors in', test_data.shape[0], 'samples')

    # Show sorted weights for every class
    for i in range(len(ws)):

        # Sort weights to find most important
        w = list(ws[i][1:])
        feature_ids = range(1, len(w) + 1)
        print('W:', w)
        labels = []
        for id in feature_ids:
            labels.append(str(int(id)))

        # Report top weights
        plt.clf()
        plt.bar(labels, w)
        plt.title('iris Perceptron: feature weights for class = ' + str(i+1))
        plt.xlabel('feature ID')
        plt.ylabel('weight')
        plt.grid(True)
        plt.savefig('iris_weights' + str(i+1) + '.png', dpi = 600)


def a4a_knn():
    """
    Run kNN on the a4a dataset for various numbers of neighbors.
    """
    print("----------\na4a kNN")

    # Load data
    train_data = libsvm_scale_import('data/a4a')
    test_data = libsvm_scale_import('data/a4a.t')

    # Training data has 1 fewer feature than test data, so add a column
    # of zeros to it so samples have same number of features in train and test
    zero_col = np.zeros((len(train_data), 1))
    train_data = np.hstack((train_data, zero_col))

    # Test multiple values of k
    # This takes over 3 hours to run on my fastest computer.
    test_ks = np.array([1, 3, 5, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 201, 301, 401, 501, 601, 701, 801, 901, 1001])
    error_rates = np.zeros(len(test_ks))
    for i in range(len(test_ks)):
        print('Classify with k =', test_ks[i])
        # Classify the test data
        classifications = k_nearest_neighbors(train_data, test_data,
                                              test_ks[i])
        # Check accuracy
        errors = check_knn_classifications(test_data[:,0], classifications)
        error_rates[i] = errors / test_data.shape[0]
        print(errors, 'errors in', test_data.shape[0], 'samples')

    print('ks:', test_ks)
    print('error rates:', error_rates)
    plt.clf()
    plt.plot(test_ks, error_rates, marker='.')
    plt.title('a4a kNN: error rate vs. k')
    plt.xlabel('k')
    plt.ylabel('error rate')
    plt.xlim(left = 0)
    plt.ylim(bottom = 0)
    plt.grid(True)
    plt.savefig('a4a_knn.png', dpi = 600)


def a4a_perceptron():
    """
    Run Perceptron on the a4a dataset in various ways.
    """
    print("----------\na4a Perceptron")

    # Load data
    train_data = libsvm_scale_import('data/a4a')
    test_data = libsvm_scale_import('data/a4a.t')

    # Training data has 1 fewer feature than test data, so add a column
    # of zeros to it so samples have same number of features in train and test
    zero_col = np.zeros((len(train_data), 1))
    train_data = np.hstack((train_data, zero_col))

    # Test multiple values of beta
    test_betas = np.array([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    error_rates = np.zeros(test_betas.shape[0])
    ws = []
    best_beta = -1
    best_error_rate = 999999
    for i in range(len(test_betas)):
        print('Classify with beta =', test_betas[i])

        # Train and find weights
        ws.append(train_perceptron(train_data, test_betas[i], 100000))

        # Check accuracy
        errors = test_perceptron(test_data, ws[i])
        error_rates[i] = errors / test_data.shape[0]
        if (error_rates[i] < best_error_rate):
            best_error_rate = error_rates[i]
            best_beta = i
        print(errors, 'errors in', test_data.shape[0], 'samples')

    # Report error rates
    print('betas:', test_betas)
    print('error rates:', error_rates)
    plt.clf()
    plt.plot(test_betas, error_rates, marker='.')
    plt.title('a4a Perceptron: error rate vs. step size for 100000 iterations')
    plt.xscale('log')
    plt.xlabel('step size')
    plt.ylabel('error rate')
    plt.ylim(bottom = 0)
    plt.grid(True)
    plt.savefig('a4a_perceptron.png', dpi = 600)

    # Sort weights to find most important
    w = list(ws[best_beta][1:])
    feature_ids = range(1, len(w) + 1)
    bar_data = list(zip(feature_ids, w))
    bar_data.sort(key = lambda _: abs(_[1]), reverse = True)
    bar_data = np.array(bar_data[:20])
    labels = []
    for id in bar_data[:,0]:
        labels.append(str(int(id)))

    # Report top weights
    plt.clf()
    plt.bar(labels, bar_data[:,1])
    plt.title('a4a Perceptron: 20 most important features')
    plt.xlabel('feature ID')
    plt.ylabel('weight')
    plt.grid(True)
    plt.savefig('a4a_weights.png', dpi = 600)


def main():
    iris_knn()
    iris_perceptron()
    a4a_knn()
    a4a_perceptron()


if __name__ == '__main__':
    main()

