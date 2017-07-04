import numpy as np

# x = np.arange(1, 10).reshape((3, 3))
# idx = np.array([0, 1, 1])
#
# print(np.choose(idx, x))
# print(x[idx, np.arange(3)])
#
# print(x)
# x[idx, np.arange(3)] = 0
# print(x)
#
# x = np.arange(1, 10).reshape((3, 3))
# print(np.sum(x))

# x = np.array([1, 2, 3, 4, 5])
# x -= 4
# print(x)
#
# x_clamp = np.fmax(0, x)
# # function np.fmax can take arrays of different size, as long as they can be broadcasted into the same size
# print(x_clamp)

#
# # about iterater index and multi-index
# a = np.arange(6).reshape(2, 3)
# it = np.nditer(a, flags=['f_index', 'multi_index'], op_flags=['writeonly'])
# while not it.finished:
#     print("<%d> <%s>" % (it.index, it.multi_index))
#     it.iternext()
# while not it.finished:
#     print(it.multi_index[1], it.multi_index[0])
#     it[0] = it.multi_index[1] - it.multi_index[0]
#     it.iternext()


# trying out global variables
# : global variables can be used directly inside a function
"""
pass-by-value: a local copy is made within the calling function to be used, so the original variable will not be changed
within the calling function (C, C++)

pass-by-reference: the reference of the original variable is passed to the function, the called variable can be chaged
within the calling function as a result

what's done here in python is a mixture:
if the value passed to the function (a in the function add_x(a)) is not changed in the function.
then it's being passed by reference.
if the value is changed, it's being passed by value, changes made to this variable has local effect only
"""
x = 5

def add_x(a):
    print(a, id(a))
    a = 10
    print(a, id(a))

add_x(x)
print(x)
