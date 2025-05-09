from scipy.io import loadmat
import numpy as np

# loading the training data
data = loadmat("mnist-original.mat")

# extracting the data
X = data['data']
X = X.transpose()

# normalize the gray scale image to scale of 0-1
X = X / 255