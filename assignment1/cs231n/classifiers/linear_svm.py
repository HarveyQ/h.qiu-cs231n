import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  """
    2 May notes:
    the gradient flow to W for
    1) the correct class is -(margin_count * x_i)
    2) all other classes is x_i if margin>0
    let's try this idea out] and then do a power nap before Copenhagen

    we also want the gradient to be divided by N and then added to the gradient
    of regularisation

    The gradient of the SVM loss, according to derivation by calculus, can be
    acquired effectively by:

    For each image (data point):
    1. count the number of classes that failed to meet the margin requirement
    2. the contribution to the loss from this image = x[i] * num_counted for all i ~= yi
    negative that for i = yi

  """

  num_classes = W.shape[1]  # == C
  num_train = X.shape[0]

  loss = 0.0
  dW = np.zeros(W.shape)  # initialise dW

  for i in xrange(num_train):
    # calculate scores
    scores = X[i].dot(W)
    scores_correct = scores[y[i]]

    margin_count = 0  # reset margin count for each image
    dW_temp = np.zeros(W.shape)  # reset dW_temp matrix for each image

    for j in xrange(num_classes):
        if j == y[i]:  # skip correct class
            continue
        margin = scores[j] - scores_correct + 1
        if margin > 0:  # max(0, margin)
            loss += margin
            margin_count += 1
            dW_temp[:, j] = X[i].T

    dW_temp[:, y[i]] = - margin_count * X[i].T
    dW += dW_temp  # add to total gradient

  # final loss and dW for return
  loss = loss/num_train + np.sum((np.power(W, 2)))  # average, add regularisation
  dW = dW/num_train + 2.0*reg*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

