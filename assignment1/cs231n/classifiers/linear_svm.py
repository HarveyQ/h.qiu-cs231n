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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]  # == C
  num_train = X.shape[0]
  loss = 0.0  # initialise lost
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    margin_count = 0  # reset margin count for each image

    for j in xrange(num_classes):
      if j == y[i]:
        continue
      # for j ~= y[i], namely the "wrong" classes
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        margin_count += 1

    # calculate the contribution to gradient for this image
    scaled_Xi = margin_count * X[i].T
    dW_temp = scaled_Xi * np.full(W.shape, 1.0)
    dW_temp[:, y[i]] = - scaled_Xi  # inverse the value before adding to overall gradient

    # add the contribution to the overall gradient
    dW += dW_temp

  """
  2 May:
  the gradient flow to W is the same magnitude except dW_yi = - sum...(x_i)
  let's try this idea out and then do a power nap before Copenhagen
  """

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  """
  we also want the gradient to be divided by N and then added to the gradient
  of regularisation
  """
  dW /= num_train

  # add regularisation to the loss
  dW += 2.0 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  """
  The gradient of the SVM loss, according to derivation by calculus, can be
  acquired effectively by:

  For each image (data point):
  1. count the number of classes that failed to meet the margin requirement
  2. the contribution to the loss from this image = x[i] * num_counted for all i ~= yi
  negative that for i = yi

  """

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
