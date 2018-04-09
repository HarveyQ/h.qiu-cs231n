# playground for cs231n Numpy tutorial
#
# # enumerate
# animals = ['cat', 'dog', 'monkey']
#
# for idx, animal in enumerate(animals):
#     print '#%d: %s' % (idx + 1, animal)


# list comprehensions
# x_list = [1, 2, 3, 4]
#
# x_sq = [x**2 for x in x_list]
# print x_sq

# library keys (are they unique?: yes they are)
# d = {'person': 2, 'fish': 3, 'lion': 4}
#
# print 'fish' in d
# print d.get('fish', 'N/A')

# animals = {'cat', 'dog', 'fish'}

# # set comprehensions
# from math import sqrt
# nums = {int(sqrt(x)) for x in range(30)}
# print nums
# print nums
#
# import numpy as np
#
# a = np.array([1, 2, 3])
# print type(a)
#
# print a.shape

import numpy as np

# # integer indexing
# a = np.array([[1,2], [3, 4], [5,6]])
# b = np.array([0, 2, 0, 1])
#
# c = a[np.arange(4), b]
# print c
#
# a[np.arange(4), b] += 10
# print a

# # using boolean array to index
# bool_idx = (a > 2)
# print bool_idx
# print a
# print a[bool_idx]
# print a[a>2]

# # data type
# x = np.array([5, 6])
# y = np.array([[5.0, 6], [7, 8]])
#
# print x, '\n', x.dtype, '\n', y, '\n', y.dtype
#
# z = np.array([5, 6], dtype=np.float64)
# print z, '\n', z.dtype

# array math
# x = np.array([[1,2],[3,4]], dtype=np.float64)
# y = np.array([[5,6],[7,8]], dtype=np.float64)
#
# print x+y
# print np.add(x, y)
#
# print np.sqrt(x)
# print np.power(x, 2)
#
# print x

# x = np.array([[1, 2], [3, 4]])
# y = np.array([[5, 6], [7, 8]])
#
# w = np.array([9, 10])
# v = np.array([11, 12])

# two equivalent method of calculating inner product between w and v
# print v.dot(w)
# print np.dot(w, v)

# print x.dot(y)
# print np.dot(x, y)

# sum function
# print np.sum(x)
# # sum of each column
# print np.sum(x, axis=0)
#
# # sum of each row
# print np.sum(x, axis=1)

x = np.array([[1, 2], [3, 4]])

# transpose
print x
print x.T

v = np.array([1, 2, 3, 4])
print v
print v.T
