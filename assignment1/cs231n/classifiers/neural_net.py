from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, output_size, hidden_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

    # store the hyperparameters as a model attribute
    # so we can track the best parameters easier
    self.hyper_params = {}
    self.hyper_params['hidden_size'] = hidden_size

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape  # N = number of examples, D = dimension of each example
    H, C = W2.shape  # H = size of the hidden layer, C = number of classes

    # Compute the forward pass
    # scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################

    # the bias trick
    W1 = np.vstack((W1, b1.reshape((1, H))))
    W2 = np.vstack((W2, b2.reshape((1, C))))

    ## staged forward path
    # 1st FC o/p
    X = np.hstack((X, np.ones((N, 1))))  # bias trick
    fc1 = np.dot(X, W1)

    # 1st ReLU o/p
    fc1_relu = np.maximum(0, fc1)

    # 2nd FC o/p
    fc1_relu = np.hstack((fc1_relu, np.ones((N, 1))))  # bias trick
    scores = np.dot(fc1_relu, W2)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    # loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################

    # calculate Softmax loss from Neural Net scores
    # *copied from Softmax classifier
    const_vec = -np.amax(scores, axis=1)  # constant for numerical stability
    scores += const_vec.reshape((N, 1))  # add constant to scores

    sum_correct_score = np.sum(scores[range(N), y])  # 1st term of data loss
    escores = np.exp(scores)
    sum_log_exp = np.sum(np.log(np.sum(escores, axis=1)))  # 2nd term of ..

    data_loss = - sum_correct_score + sum_log_exp  # data loss
    reg_loss = np.sum(np.power(W1, 2)) + np.sum(np.power(W2, 2))

    loss = (1/N) * data_loss + reg * reg_loss  # final loss

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    ## staged backprop: calculate gradient from data loss to weights W1 and W2
    # into Softmax loss: dL/dfc2
    sum_escores_inv = 1.0/np.sum(escores, axis=1)
    dfc2 = escores * sum_escores_inv.reshape((len(sum_escores_inv), -1))  # multiply with broadcast
    dfc2[range(N), y] += -1
    dfc2 /= N

    # into FC2: dL/dW2 and dL/dReLU
    dW2 = np.dot(fc1_relu.T, dfc2)  # into W2
    dReLU = np.dot(dfc2, W2[:-1].T)

    # into ReLU: dL/dfc1
    relu_mask = fc1 < 0
    dfc1 = dReLU
    dfc1[relu_mask] = 0  # kill gradient at fc<0

    # into FC1 (with the bias trick): dL/dW1
    dW1 = np.dot(X.T, dfc1)

    # include gradient from L2 regularisation
    dW1 += 2 * reg * W1
    dW2 += 2 * reg * W2

    ## unpack gradients
    grads['W1'] = dW1[:-1]
    grads['W2'] = dW2[:-1]

    grads['b1'] = dW1[-1]
    grads['b2'] = dW2[-1]

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_epochs=1,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - (deprecated) num_iters: Number of steps to take when optimizing.
    - num_epochs: Number of training epochs
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.

    Return:
    A dictionary of records:
    {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }
    """

    # store the hyperparameters in the model itself
    self.hyper_params['learning_rate'] = learning_rate
    self.hyper_params['num_epochs'] = num_epochs
    self.hyper_params['reg'] = reg
    # self.hyper_params['drop'] = 1  # future: drop out

    num_train = X.shape[0]
    iterations_per_epoch = int(max(num_train / batch_size, 1))
    num_iters = num_epochs * iterations_per_epoch

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      batch_idx = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[batch_idx]
      y_batch = y[batch_idx]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      # update first FC layer
      self.params['W1'] += - learning_rate * grads['W1']
      self.params['b1'] += - learning_rate * grads['b1']

      # update hidden layer
      self.params['W2'] += - learning_rate * grads['W2']
      self.params['b2'] += - learning_rate * grads['b2']

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and (it+1) % 10 == 0:  # HQ: display info every 100 iterations
        print('iteration %d / %d: loss %f' % (it+1, num_iters, loss))
        print('W2 sum: %f' % np.sum(self.params['W2']))
        print('W1 sum: %f' % np.sum(self.params['W1']))

      # Every epoch, check train and val accuracy and decay learning rate.
      if (it+1) % iterations_per_epoch == 0:
        if verbose:
          print('number of epochs completed: %d' % ((it+1) / iterations_per_epoch))

        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate (annealing)
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history
      }

  def forward_path_once(self, X):
    """
    staged forward path packed in one function

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Outputs:
    - scores: a numpy array of shape (N, C), scores calculated from forward path
    """

    # unpack parameters
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    # some dimensions
    N, D = X.shape  # N = number of examples, D = dimension of each example
    H, C = W2.shape  # H = size of the hidden layer, C = number of classes

    # staged forward path
    # 1st FC o/p
    W1 = np.vstack((W1, b1.reshape((1, H))))  # bias trick
    X = np.hstack((X, np.ones((N, 1))))
    fc1 = np.dot(X, W1)

    # 1st ReLU o/p
    fc1_relu = np.maximum(0, fc1)

    # 2nd FC o/p
    W2 = np.vstack((W2, b2.reshape((1, C))))  # bias trick
    fc1_relu = np.hstack((fc1_relu, np.ones((N, 1))))
    scores = np.dot(fc1_relu, W2)

    return scores

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################

    scores = self.forward_path_once(X)
    y_pred = np.argmax(scores, axis=1)  # predict as the class with highest score

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


