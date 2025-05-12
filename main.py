from scipy.io import loadmat
import numpy as np
import randomInitialize

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
