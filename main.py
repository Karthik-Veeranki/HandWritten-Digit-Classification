from scipy.io import loadmat
from scipy.optimize import minimize
import numpy as np
import randomInitialize
import neuralNetwork

# loading the training data
# mnist dataset is the handwritten digits data set containing pixel data of images:
# Images data : 784 x 70,000
# Labels data : 1 x 70,000
data = loadmat("mnist-original.mat")

# extracting the data
X = data['data']
X = X.transpose()

# normalize the gray scale image to scale of 0-1
X = X / 255

y = data['label']
y = y.flatten()

# split the data into 60,000 training samples and 10,000 testing samples
X_train, y_train = X[:60000, :], y[:60000]       # index 0 - 59,999
X_test, y_test = X[60000:, :], y[60000:]         # index 60,000 - 79,999

input_layer_size = 784        # 28x28 pixel data, hence 784 features
hidden_layer1_size = 100      # first fully connected layer size = 100
hidden_layer2_size = 50       # second fully connected layer size = 50
num_of_labels = 10            # final layer size = number of prediction classes


# randomly initializing weights and bias for each of the layers
initial_weights_1 = randomInitialize.randomInitialize(hidden_layer1_size, input_layer_size)
initial_weights_2 = randomInitialize.randomInitialize(hidden_layer2_size, hidden_layer1_size)
initial_weights_3 = randomInitialize.randomInitialize(num_of_labels, hidden_layer2_size)

# flattening the parameters into single vector
initial_nn_parameters = np.concatenate((
    initial_weights_1.flatten(),
    initial_weights_2.flatten(),
    initial_weights_3.flatten()
))

max_iter = 100
lambda_reg = 0.1
my_args = (X_train, y_train, input_layer_size, hidden_layer1_size, hidden_layer2_size, num_of_labels, lambda_reg)

# Calling minimize function to optimize the cost function
final_results = minimize(neuralNetwork.neuralNetwork, x0=initial_nn_parameters,  # function output which we need to optimize
                         args=my_args,                                           # arguments passed for the function, other than initial parameters.
                         options={'maxiter': max_iter},                          # setting stopping criteria as iterations <= 100
                         method='L-BFGS-B',                                      # Limited memory optimization method used mainly for large-scale computations
                         jac=True)                                               # setting jacobian method of computing gradient vector.

