# -*- coding: utf-8 -*-
"""
Neural Network homework

Dennis Brown, COMP6636, 11 APR 2021
"""

import numpy as np


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


# Test it out

# Use the truth table for HW problem #1
print('Train on this:')
X = np.zeros([16, 4])
y = np.zeros([16, 2])
print('X             Y')
for i in range(16):
    xbool = [(j == '1') for j in bin(i)[2:].zfill(4)]
    X[i] = np.array(xbool) * 1
    ybool = (xbool[3] or (not xbool[2])) ^ ((not xbool[1]) and (not xbool[0]))
    y[i][0 if ybool else 1] = 1
    print(X[i], y[i])

# Train the neural network
xh, hy = train_neural_network(X, y, 4, 0.5, 10000)
print('H weights')
print(xh)
print('Output weights')
print(hy)

# Test the neural network
print('Test it out:')
y_hat = test_neural_network(X, xh, hy)
y_hat = 1.0 * (y_hat > 0.5)
print(y_hat)
print('Testing error:')
print(y - y_hat)


