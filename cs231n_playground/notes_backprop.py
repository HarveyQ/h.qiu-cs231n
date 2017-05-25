import numpy as np
import math

"""backprop through a toy example expression"""

x = 3
y = -4

"""forward pass"""
sigy = 1.0 / (1 + math.exp(y))
num = x + sigy  # numerator
sigx = 1.0 / (1 + math.exp(x))
xpy = x + y
xpysqr = xpy ** 2
den = sigx + xpysqr  # denominator
invden = 1.0 / den  # inverse of the denominator (stage for calculus purpose)
f = num * invden


"""backprop"""
# into the num
dnum = invden

dx = 1.0 * dnum
dsigy = 1.0 * dnum

dy = dsigy * (1 - sigy) * sigy

# into the denominator
dinvden = num
dden = -1.0 * (den**2) * dinvden

# into sigx
dsigx = 1.0 * dden
dx += (1 - sigx) * sigx * dsigx

# into (x+y)**2
dxpysqr = 1.0 * dden
dxpy = 2.0 * dxpysqr

dx += 1.0 * dxpy
dy += 1.0 * dxpy

# Done!

"""
notice the += instead of =
gradients from branches of a variable flow back and add up
"""

print(x, y, dx, dy)
