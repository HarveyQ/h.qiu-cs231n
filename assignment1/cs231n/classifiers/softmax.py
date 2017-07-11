import numpy as np
from random import shuffle
from past.builtins import xrange
import math

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]  # N
  num_class = W.shape[1]  # C
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  # HQ: numerical instability problem is countered by using the c constant    #
  #############################################################################
  for i in xrange(num_train):
    # calculate loss of this data point
    xi = X[i]
    scores = xi.dot(W)  # calculate score of data point Xi
    const = math.exp(-np.amax(scores))  # constant for numerical stability
    e_scores = np.exp(scores + const)
    Li = -math.log(e_scores[y[i]]/(np.sum(e_scores)))
    loss += Li

    # calculate the gradient of this data point (analytical)
    idx_yi = np.zeros((1, num_class))
    idx_yi[0, np.arange(num_class) == y[i]] = 1
    Myi = np.outer(xi, idx_yi)  # term 1
    num = np.outer(xi, e_scores)  # term 2: numerator
    den = np.sum(e_scores)  # term 2: denominator
    dWi = -Myi + num/den  # assemble gradient
    dW += dWi  # accumulate gradient

  loss_reg_term = reg * np.sum(np.power(W, 2))  # calculate regularisation term
  loss = loss/num_train + loss_reg_term  # finalise the loss

  dW = dW/num_train + 2*reg*W  # finalise the gradient

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  # get some dimensions
  num_train = X.shape[0]  # N
  num_class = W.shape[1]  # C
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################


  # loss =
  # dW =
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
