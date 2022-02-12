import torch 
import numpy as np
from scipy.signal import convolve2d, correlate2d
from splatter import splat_conv2d, splat_corr2d



for i in range(2, 25):
    for k in range(2,25):

        image = torch.randint(low=0, high=255, size=(i,i), dtype=torch.uint8)
        kernel = torch.randint(low=0, high=5, size=(k,k))


        image = image.numpy()

        # print(kernel)
        kernel = kernel.numpy()
        image_base = convolve2d(image, kernel, mode= "valid")

        image_splat = splat_conv2d(image, kernel, mode="valid")

        # print(image_base)

        # print()
        # print(image_splat)

        if(np.array_equal(image_base, image_splat)):
            print("yay")
        else:
            print("boo", i, k)


# i = 2
# k = 3
# image = torch.randint(low=0, high=255, size=(i,i), dtype=torch.uint8)
# kernel = torch.randint(low=0, high=5, size=(k,k))


# image = image.numpy()

# # print(kernel)
# kernel = kernel.numpy()
# image_base = convolve2d(image, kernel, mode= "valid")

# image_splat = splat_conv2d(image, kernel, mode="valid")

# print(image_base)

# print()
# print(image_splat)

# if(np.array_equal(image_base, image_splat)):
#     print("yay")
# else:
#     print("boo", i, k)