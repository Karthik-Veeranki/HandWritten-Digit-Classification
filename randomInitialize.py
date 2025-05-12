import numpy as np

'''
For initializing the weights and bias matrix, we use the method of 
            "UNIFORM XAVIER GLOROT INITIALIZATION"

The idea is to create the weights matrix of size x * (y+1)
        +1 accounts for the bias term in each row

Initialize each value in the range of [-epsilon, epsilon]
    where epsilon = sqrt(6/x+y)

By this method, we tackle the following problems:
1) Vanishing gradient problem, if the value becomes extremely small, slowing down the learning
2) Exploding gradient problem, if the value becomes extremely large, causing unstable divergent model.
'''


def randomInitialize(x, y):
    epsilon = np.sqrt(6) / np.sqrt(x + y)
    return np.random.rand(x, y + 1) * (2 * epsilon) - epsilon

#                           |              |             |
# size of weights matrix-----              |             |
#                                          |             |
# random init in range [0,2*epsilon]--------             |
#                                                        |
# scaling down the range to [-epsilon, +epsilon]----------