# practice codes in the lecture notes
import numpy as np

# SVM loss
def svm_L_i_vectorised(x, y, W):
    """
    half-vectorised implementation
    the function calculates the loss for each example without loop
    """
    delta = 1.0
    scores = W.dot(x)

    margins = np.maximum(0, scores - scores[y] + delta)
    # ignore the margin on the i-th position since it's the correct class
    margins[y] = 0
    loss_i = np.sum(margins)

    return loss_i


def svm_L(X, y, W):
    """
    fully-vectorized implementation :
    - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
    - y is array of integers specifying correct class (e.g. 50,000-D array)
    - W are weights (e.g. 10 x 3073)
    """
    delta = 1.0
    scores = W.dot(X)
    # a row array of y-th score of each example (column), making use of broadcasting
    scores_y = scores[y, np.arange(scores.shape[1])]

    margins = np.fmax(0, scores - scores_y + delta)
    margins[y, np.arange(scores.shape[1])] = 0
    loss = np.sum(margins)

    return loss
