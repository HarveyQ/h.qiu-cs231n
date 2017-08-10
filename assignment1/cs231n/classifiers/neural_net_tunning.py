import numpy as np
import copy
from cs231n.classifiers.neural_net import TwoLayerNet

"""
HQ note:
some pretty useful functions that I wrote for tunning of two layer neural net
"""

def hyper_params_comb(hyper_params_range):
    """
    A thin wrapper for the combination generator

    Input:
    - hyper_params_range: a list of the ranges (sublists) of hyperparameters

    Output:
    - hyper_params_list: a list of combinations of hyperparameters

    Future notes:
    - This method is hardly scalable or efficient for models with more hyperparams
    Keep looking for scalable methods for the same purpose.
    e.g. try and integrate the np.meshgrid() method

    """
    hyper_params_list = []
    for hs in hyper_params_range[0]:
        for lr in hyper_params_range[1]:
            for ne in hyper_params_range[2]:
                for reg in hyper_params_range[3]:
                    hyper_params_list.append([hs, lr, ne, reg])
    return hyper_params_list


def net_tuning(X_train, y_train, X_val, y_val, hyper_params_list, verbose=False):
    """
    Inputs:
    - X_train: numpy array of shape (num_train, D) training data
    - y_train: numpy array of shape (num_train, ) traning labels
    - X_val: numpy array of shape (num_val, D) validation data
    - y_val: numpy array of shape (num_val, ) validation data
    - hyper_params_list: list of hyperparameter combinations
    - verbose: display some annoying progress info if True

    Outputs:
    - best_net: the best model found within all combinations (deep copied)
    - resluts: a dictionary {(hyper_params): (train_accuracy, val_accuracy)}
    """

    input_size = X_train.shape[1]
    num_classes = 10  # warning: hard coded number (CIFAR-10)

    results = {}
    best_val_acc = 0
    best_net = None

    hyper_params_count = 0
    for idx in range(len(hyper_params_list)):
        # unpack hyperparameters
        hyper_params = hyper_params_list[idx]
        hidden_size = hyper_params[0]
        learning_rate = hyper_params[1]
        num_epochs = hyper_params[2]
        reg = hyper_params[3]

        # train network
        net = TwoLayerNet(input_size, num_classes, hidden_size)
        stats = net.train(X_train, y_train, X_val, y_val,
                          batch_size=200, learning_rate_decay=0.95,
                          num_epochs=num_epochs, learning_rate=learning_rate, reg=reg,
                          verbose=verbose)

        # predict and calculate accuracy
        train_acc = (net.predict(X_train) == y_train).mean()
        val_acc = (net.predict(X_val) == y_val).mean()
        results[tuple(hyper_params)] = (train_acc, val_acc)  # pack results

        # update best model and best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_net = copy.deepcopy(net)  # store the best net model

        hyper_params_count += 1
        print('Hyperparameter combinations completed: %d / %d' % (hyper_params_count, len(hyper_params_list)))
        print()

    print('The best validation accuracy acheived: %f' % best_val_acc)

    print()
    print('Hyperparamters of the best net:')
    print('hidden size: %d' % best_net.hyper_params['hidden_size'])
    print('learning rate: %f' % best_net.hyper_params['learning_rate'])
    print('number of epochs: %d' % best_net.hyper_params['num_epochs'])
    print('regularisation strength: %f' % best_net.hyper_params['reg'])
    print()

    return best_net, results
