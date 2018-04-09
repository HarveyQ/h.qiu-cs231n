import numpy as np

# # slicing
# a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# b = a[:2]
# print a.shape
# print b
# # e = np.eye(2)
# # print e

# # modifying a sub-version of "x" also modified the original array
# x = np.array([[1,2,3,4], [5,6,7,8]])
# y = x[:, 1:3]
# print y
#
# y[0, 0] = 10
# print y
# print x

#
# # integer indexing lowers the rank of the array in that dimension
# x = np.array([[1,2,3,4], [5,6,7,8]])
# y1 = x[1, 1:]  # indexing with integer in x dimension
# y2 = x[1:2, 1:]  # indexing without integer in both dimensions
# print y1, y1.shape  # y1 is rank 1
# print y2, y2.shape  # y2 is rank 2


# # integer array indexing
# x = np.array([[1,2,3], [4,5,6], [7,8,9]])
# subx = x[[0, 1, 2, 2], [0, 1, 2, 0]]  # the indexing is in the form of x[[x1, x2, x3, x4...], [y1, y2, y3, y4, ...]]
# print subx

# integer array indexing trick
# a = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])

# a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
# print a
#
# b = np.array([0, 2, 0, 1])
# print a[np.arange(4), b]  # select one element from each row using b
#
# a[np.arange(4), b] += 10  # mutate one element from each row of a using b
# print a


# # Boolean array indexing
# a = np.array([[1,2], [3, 4], [5, 6]])
# bool_idx = (a > 2)
# print bool_idx
# print a[bool_idx]
# print a[a > 2]  # equivalently

#
# # datatypes
# # elements in one numpy array has the same datatype
# x = np.array([1, 2])
# print x.dtype  # int64
#
# x = np.array([1.0, 2.0])
# print x.dtype  # float64
#
# x = np.array([1.0, 2.0], dtype=np.int64)  # forcing a datatype
# print x
# print x.dtype
#

#
# # array math
# x = np.array([[1, 2], [3, 4]], dtype=np.float64)
# y = np.array([[5, 6], [7, 8]], dtype=np.float64)

# print x+y
# print np.add(x, y)  # add
#
# print x-y
# print np.subtract(x, y)
#
# print x*y
# print np.multiply(x, y)  # element-wise multiplication
# print np.dot(x, y)  # inner product 1
# print x.dot(y)  # inner product 2
# print x.dot(y).shape  # matrix . matrix = rank2 results
# print x.dot(np.arange(2)).shape  # matrix . vector = rank 1

# print x/y
# print np.divide(x, y)
#
# print np.sqrt(x)
#
#
#
# # sum
# x = np.full((3, 4), 5, dtype=np.int64)
# print x
#
# y = np.sum(x, axis=0)  # sum of each column
# print y, y.shape
#
# z = np.sum(x, axis=1)  # sum of each row
# print z, z.shape
#
# print np.sum(x)  # total sum (axis undefined)
#
#
# # transpose of matrix
# print x.shape
# print x.T, x.T.shape
#         # transposing a rank 1 matrix (vector) has no effect


# # Broadcasting
# x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
# v = np.array([1, 0, 1])
# y = x + v  # add v to each row of x using broadcasting
# print y
#
# # the broadcasting is equivalent to:
# vv = np.tile(v, (4, 1))
# y2 = x + vv
# print y2==y
#
# # example using broadcasting
# v = np.array([1, 2, 3])
# w = np.array([4, 5])
#
# vv = np.reshape(v, (3, 1))
# v_outer_w = vv * w  # outer product with broadcasting
# print v_outer_w

