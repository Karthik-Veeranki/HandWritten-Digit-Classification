import numpy as np
import lossFunctions

'''
This function performs the overall neural network operations.
It takes the following parameters:

1) X_train               - Training data
2) y_train               - Training labels
3) initial_nn_parameters - flattened vectors containing initial weights for all the layers in NN
4) input_layer_size      - size of input data (28 x 28)
5) hidden_layer1_size    - size of first hidden layer
6) hidden_layer2_size    - size of second hidden layer
7) num_of_labels         - number of labels in final layer (0,1,...,9)
8) lambda_reg            - regularization parameter to prevent overfitting

Outputs the following results:
1) cost                  - cost function
2) gradient              - gradient vector

'''


def neuralNetwork(X_train, y_train, initial_nn_parameters, input_layer_size, hidden_layer1_size, hidden_layer2_size, num_of_labels, lambda_reg):
    # initial_nn_parameters are split back into weights matrices
    lim1 = hidden_layer1_size * (input_layer_size + 1)
    lim2 = hidden_layer2_size * (hidden_layer1_size + 1)

    weights_1 = np.reshape(initial_nn_parameters[0:lim1],
                           (hidden_layer1_size, input_layer_size + 1))
    weights_2 = np.reshape(initial_nn_parameters[lim1 : (lim1 + lim2)],
                           (hidden_layer2_size, hidden_layer1_size + 1))
    weights_3 = np.reshape(initial_nn_parameters[(lim1 + lim2):],
                           (num_of_labels, hidden_layer2_size + 1))
    
    # Forward propagation
    # At each of the layers, the computation of sigma(WX + b) is taken
    # Out of previous layer is given as input to the next layer

    m = X_train.shape[0]
    ones_matrix = np.ones((m, 1))

    # input layer --------> first hidden layer
    a1 = np.append(ones_matrix, X_train, axis=1)
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

    # ONE HOT ENCODING
    # For ease of calculation, we convert the y_train into a 2D matrix y_calc
    # Size = no_of_samples x 10
    # ycalc[i][j] = {1, if y_train[i] = j}

    y_one_hot_enc = np.zeros((m, 10))
    for i in range(m):
        j = int(y_train[i])
        y_one_hot_enc[i][j] = 1
    

    # COST FUNCTION CALCULATION
    # Total cost = Cross entropy loss + L2 Regularization loss
    cost = lossFunctions.crossEntropyLoss(m, y_one_hot_enc, a4) + lossFunctions.L2RegLoss(lambda_reg, m, weights_1, weights_2, weights_3)

    
    pass