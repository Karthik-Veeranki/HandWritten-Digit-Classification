import numpy as np

def predict(weights_1, weights_2, weights_3, X_test):
    m = X_test.shape[0]
    ones_matrix = np.ones((m, 1))

    # input layer --------> first hidden layer
    a1 = np.append(ones_matrix, X_test, axis=1)
    z2 = np.dot(a1, weights_1.transpose())
    a2 = 1 / (1 + np.exp(-z2))

    # first hidden layer ----------> second hidden layer
    a2 = np.append(ones_matrix, a2, axis=1)
    z3 = np.dot(a2, weights_2.transpose())
    a3 = 1 / (1 + np.exp(-z3))

    # second hidden layer ----------> output layer
    a3 = np.append(ones_matrix, a3, axis=1)
    z4 = np.dot(a3, weights_3.transpose())
    a4 = 1 / (1 + np.exp(-z4))

    pred = (np.argmax(a4, axis=1))
    return pred