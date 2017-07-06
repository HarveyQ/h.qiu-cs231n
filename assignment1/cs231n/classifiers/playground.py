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

import numpy as np
# x = np.array([[1,2,3],[4,5,6],[7,8,9]])
# x1 = x-5
#
# c = np.zeros((3,))
# c[1] = -np.sum(x1[1]>0)
# print(x1)
# print(c)


x1 = np.array([1, 2, 3, 4, 5, 6])
x2 = np.array([5, 0, 1])

m1 = np.outer(x1, x2)
m2 = x1.reshape((x1.shape[0], -1)) * x2.reshape((-1, x2.shape[0]))


print(str(m1) + '\n')
print(str(m2))
print(np.sum(m1 != m2))


