import numpy as np

# results = {}

learning_rates = [1e-7, 5e-5]
regularization_strengths = [2.5e4, 5e4]


num_lr = len(learning_rates)
num_reg = len(regularization_strengths)
hyper_para = np.zeros((num_lr * num_reg, 2))
idx = 0
for lr in learning_rates:
    for reg in regularization_strengths:
        hyper_para[idx] = np.array([lr, reg])
        idx += 1
print(hyper_para)

# num_lr = len(learning_rates)
# num_reg = len(regularization_strengths)
# idx = 0
# hyper_para = np.zeros((num_lr * num_reg, 2))
# for idx_lr, lr in enumerate(learning_rates):
#     for idx_reg, reg in enumerate(regularization_strengths):
#         hyper_para[idx] = np.array([lr, reg])
#         idx += 1
#
# print(hyper_para)
# for (lr, reg) in hyper_para:
#     results[(lr, reg)] = [10, 10]


# for lr, reg  in sorted(results):
#     train_acc, val_acc = results[(lr, reg)]
#     print('%e, %e, %f, %f' % (lr, reg, train_acc, val_acc))



# class test_class(object):
#     def __init__(self):
#         self.y = None
#
#     def func1(self, x):
#         self.y = x
#
# tc = test_class()
# tc.func1(10)
# print(tc.y)




# import numpy as np
# from past.builtins import xrange
#
# learning_rates = [1e-7, 5e-5]
# regularization_strengths = [2.5e4, 5e4]
#
# num_lr = len(learning_rates)
# num_reg = len(regularization_strengths)
# hyper_para = np.zeros((num_lr * num_reg, 2))
#
# for idx_lr, lr in enumerate(learning_rates):
#     for idx_reg, reg in enumerate(regularization_strengths):
#         idx = idx_lr + idx_reg
#         hyper_para[idx] = np.array([lr, reg])
#
# # print(hyper_para)
#
#
# for (lr, reg) in hyper_para:
#     pass
#
# print(lr, reg)

# from random import randrange
# import numpy as np
# # from past.builtins import xrange
#
# ix = tuple([randrange(m) for m in x.shape])
#
# for i in range(10):
#     print(ix)
# x = np.random.randn(3072, 10)
#

# import numpy as np
# # x = np.array([[1,2,3],[4,5,6],[7,8,9]])
# # x1 = x-5
# #
# # c = np.zeros((3,))
# # c[1] = -np.sum(x1[1]>0)
# # print(x1)
# # print(c)
#
#
# x1 = np.array([1, 2, 3, 4, 5, 6])
# x2 = np.array([5, 0, 1])
#
# m1 = np.outer(x1, x2)
# m2 = x1.reshape((x1.shape[0], -1)) * x2.reshape((-1, x2.shape[0]))
#
#
# print(str(m1) + '\n')
# print(str(m2))
# print(np.sum(m1 != m2))


