import numpy as np


def crossEntropyLoss(m, y_one_hot_enc, a4):
    loss = (-1 / m) * (np.sum(np.sum((y_one_hot_enc * np.log(a4)) + (1 - y_one_hot_enc) * np.log(1 - a4))))
    return loss


def L2RegLoss(lambda_reg, m, weights_1, weights_2, weights_3):
    # in order to keep the calculation in derivative simple, we divide by a factor of 2*m
    # Also, we don't need the bais term while calculating L2 loss. Hence, we remove it

    loss = (lambda_reg / (2 * m)) * (
        np.sum(np.sum(pow(weights_1[:, 1:], 2))) +
        np.sum(np.sum(pow(weights_2[:, 1:], 2))) +
        np.sum(np.sum(pow(weights_3[:, 1:], 2)))
    )
    
    return loss