import numpy as np

# experimenting with meshgrid
y = np.arange(0, 3, 1)
x = y.copy()
z = np.array([5])
grid = np.meshgrid(x, y, z, indexing='xy')

for ix in range(len(x)):
    for iy in range(len(y)):
        for iz in range(len(z)):
            print(grid[0][ix, iy, iz], grid[1][ix, iy, iz], grid[2][ix, iy, iz])

print(grid[0].shape)

# z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
# h = plt.contou        rf(x,y,z)
# plt.show()
#
# def test_func(x, y):
#     return x+y+z
# 
# x = 2
# y = 3
# z = 4
# 
# print(test_func(x, y))
# 



# # trying the lambda function
# def full_function(input1, input2):
#     x = input1 * input2
#     y = input1 + input2
#     return x, y
#
# input2 = 3
# lf1 = lambda input1: full_function(input1, input2)
# print(lf1(input1=2))
#
#
# lf2 = lambda input_tuple: full_function(input_tuple[0], input_tuple[1])
# print(lf2((2, 3)))

'''
PyCharm remote deployment testing

# 2017-07-17 testing PyCharm remote deployment:
making a change in local project 


# 2017-07-18:
Test 1:
making a change on DOWNLOADED file
sync, check the remote file

Test 2:
making change on remote host
sync, check the local file for this change


# 2017-07-19:
Test 3:
change on local, auto-save upload upon save?
Result: yes

Test 4:
auto-upload diabled, make a change and save, upload?
Result: no, has to be done manually

another try?

'''

# # results = {}
#
# learning_rates = [1e-7, 5e-5]
# regularization_strengths = [2.5e4, 5e4]
#
#
# num_lr = len(learning_rates)
# num_reg = len(regularization_strengths)
# hyper_para = np.zeros((num_lr * num_reg, 2))
# idx = 0
# for lr in learning_rates:
#     for reg in regularization_strengths:
#         hyper_para[idx] = np.array([lr, reg])
#         idx += 1
# print(hyper_para)

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


