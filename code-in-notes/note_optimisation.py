"""
1. this note can not be evaluated and executed straight away, different sections of this code does not streamline
2. The first method used to compute gradient in this script is similar to Andrew Ng's "gradient checking",
    This method is very computational heavy for a CPU. In this script, we used only 1000 samples from the training
    dataset just to demonstrate code validation
"""
import numpy as np
import random
from cs231n.data_utils import load_CIFAR10
import notes_loss_functions as lossfunc

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# setting up the dataset
num_training = 1000
num_validation = 1000
num_test = 1000
num_dev = 500

mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# reshape every image in X_train so all RGB data is in one vector
# then stack with 1 as the bias trick (WX+b -> WX)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_train = X_train.T

# # 1: random search
# bestloss = float("inf")
# for num in range(1000):
#     W = np.random.randn(10, 3073) * 0.0001
#     loss = lossfunc.svm_L(X_train, y_train, W)
#     if loss < bestloss:
#         bestloss = loss
#         bestW = W
#     print('in attempt %d the loss was %f, best %f' % (num, loss, bestloss))

# 2: random local search
# 3: gradient descent

# computing the gradient numerically
def eval_numerical_gradient(f, x):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array - vector) to evaluate the gradient at
    """
    fx = f(x)
    grad = np.zeros(x.shape)
    h = 0.00001  # a tiny number, representing the lim h->0

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h
        fxh = f(x)  # evaluate f[x+h] at this dimension
        x[ix] = old_value  # restore the previous value before add increment on the next dimension

        grad[ix] = (fxh - fx)/h  # calculate the gradient
        it.iternext()  # go to the next dimension

    return grad


# calculating the loss function
def CIFAR10_loss_fun(W):
    return lossfunc.svm_L(X_train, y_train, W)

W = np.random.rand(10, 3073) * 0.001
# evaluate the gradient of the function CIFAR10_loss_fun @ random weight W
df = eval_numerical_gradient(CIFAR10_loss_fun, W)


### vanilla version of Gradient Descent
# this section is not executable
while True:
    weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += - step_size * weights_grad  # parameter update


# Vanilla Minibatch Gradient Descent
while True:
    data_batch = sample_training_data(data, 256)  # sample 256 examples in each batch
    weights_grad = evaluate_gradient(loss_func, data_batch, weights)
    weights += - step_size * weights_grad