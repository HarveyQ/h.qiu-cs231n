# # list
# animals = ['cat', 'dog', 'bird']
#
# for idx, animal in enumerate(animals):
#     print '#%d: %s' % (idx+1, animal)

# # list comprehensions:
# nums = [0, 1, 2, 3, 4]
# squares = [x**2 for x in nums]
# print squares
#
# # list comprehension with conditions:
# nums = [0, 1, 2, 3, 4]
# sq_cond = [x**2 for x in nums if 1 < x < 4]
# print sq_cond
#
# # dictionaries
# d = {'cat': 'cute', 'dog': 'furry'}  # key:value pairs as entries
# print d['cat']  # get an entry by key
# print 'cat' in d  # check if 'cat' is a key in this dictionary
# d['fish'] = 'wet'  # add a pair of entry
# print d['fish']
# print d.get('monkey', 'N/A')
# del d['fish']  # remove an element
# print d.get('fish', 'N/A')
# # d['fish'] gets a runtime error instead of 'N/A"
#
# # loops dictionary
# d = {'person': 2, 'cat': 4, 'spider': 8}
# for animal in d:  # direct iteration on keys
#     legs = d[animal]
#     print 'A %s has %d legs' % (animal, legs)
#

# # #method2
# d = {'person': 2, 'cat': 4, 'spider': 8}
# for animal, legs in d.iteritems():
#     print 'A %s has %d legs' % (animal, legs)
# # dictionary.iteritems() returns pair (key, item) in iteration
#
#
# # dictionary comprehensions: used for construct a dictionary
# nums = [0, 1, 2, 3, 4]
# even_num_to_square = {x: x**2 for x in nums if x % 2 == 0}
# print even_num_to_square

# sets, is an unordered list (without index)
# animals = {'cat', 'dog', 'fish', 'lion', 'tiger'}
# print 'cat' in animals
# print 'fish' in animals
# animals.add('fish')
# animals.remove('cat')
# print animals

# for idx, animal in enumerate(animals):
#     print 'No. %d: %s' % (idx + 1, animal)
#
# # using comprehension to build a set
# from math import sqrt
# nums = {int(sqrt(x)) for x in range(30)}
# print nums


# # Tuples: immutable ordered list of values
# # tuple can be used as keys in dictionary and element of sets
# d = {(x, x + 1): x for x in range(10)}  # tuples are used as keys
# t = (5, 6)  # tuple
# print type(t)
# print d[t]
# print d[(1, 2)]
#
# l = {(x, x+1) for x in range(10)}

# Functions
# Case1: now we build a function that determines the sign of a number
# def sign(x):
#     if x > 0:
#         return 'positive'
#     elif x < 0:
#         return 'negative'
#     else:
#         return 'zero'
#
# # now let's use the function we just built
# x_list = [x-5 for x in range(10)]
# for x in x_list:
#     print sign(x)

# # Case2: define a function to take optional keyword
# def hello(name, long=False):
#     if long:
#         print 'HELLOOOOO, %s!' % name.upper()
#     else:
#         print 'Hello, %s!' % name
#
# name_list = {'Bob': True, 'Tom': False, 'Harvey': True}
# for name, loud in name_list.iteritems():
#     hello(name, loud)

# # Classes
# class Greeter(object):
#
#     # constructor
#     def __init__(self, name):
#         self.name = name  # create an instance
#
#     # Instance method
#     def greet(self, loud=False):
#         if loud:
#             print 'Hello, %s' % self.name.upper()
#         else:
#             print 'Hello, %s' % self.name
#
#     def hug(self, tight=False):
#         if tight:
#             print 'Wow strong hug, %s!' % self.name
#         else:
#             print 'Good hug, %s!' % self
#
# # use this class now
# g = Greeter('Fred')  # create an instance
# g.greet()
# g.greet(loud=True)
# g.hug(tight=True)

###############################################################
# that's it for python in general
