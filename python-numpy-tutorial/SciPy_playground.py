# SciPy offers packages to handle basic image operations
from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt

img = imread('assets/cat.jpg')
print img.dtype, img.shape

img_tinted = img * [1, 0.95, 0.9]  # scale blue, green channel by 0.95 and 0.9 respectively
img_tinted = imresize(img_tinted, (300, 300))  # resize the image to be square
imsave('assets/cat_tinted.jpg', img_tinted)

plt.close('all')

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(img_tinted)
plt.title('Tinted')
plt.show()
