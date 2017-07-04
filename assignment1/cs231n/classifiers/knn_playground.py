import numpy as np

class test(object):

    def __init__(self):
        pass

    def train(self, x):
        self.xtrain = x

    @staticmethod
    def printfun(p):
        print('printing for fun', p)

    def label_vote(self, closest_y):
        """
        Find predicted label from closest k labels by majority voting
        Settle ties by choosing the smaller label

        Input:
         - closest_y: array of shape (k,) with the label of k closest training points
        Returns:
         - label_pred: the predicted label
        """
        label_pool = np.unique(closest_y)
        max_count = 0
        label_pred = 0
        for l in label_pool:
            label_count = np.sum(closest_y == l)
            if label_count > max_count:
                max_count = label_count
                label_pred = l
            elif label_count == max_count:
                if l < label_pred:
                    label_pred = l
        self.printfun(10)
        return label_pred

# testing this function

test_inst = test()

y = np.array([1, 3, 3, 3, 2, 2, 2, 3, 2])
l_pred = test_inst.label_vote(y)
print(l_pred)
