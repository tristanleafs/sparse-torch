import scipy
import torch 
import numpy as np
from scipy.signal import convolve2d, correlate2d
from splatterUpdate import splat_conv2d, splat_corr2d
from call_splatter import splatter_backward_input_full, splatter_backward_filter_full, splatter_backward_filter, splatter_forward
import time

# input = np.random.rand(65536)
# input = input.reshape((256,256))
# kernel = np.random.rand(9)
# kernel = kernel.reshape((3,3))

input = torch.rand(1000, 1000, dtype = torch.float64).numpy()
kernel = torch.rand(3,3, dtype=torch.float64).numpy()

# print(input)

t0 = time.perf_counter()
output1 = convolve2d(input, kernel, mode="valid")
t1 = time.perf_counter()

vanilla_time = t1-t0

t0 = time.perf_counter()
output2 = splatter_forward(input, kernel)
t1 = time.perf_counter()

c_time = t1-t0



# t0 = time.perf_counter()
# output2 = correlate2d(input, kernel, mode="valid")
# t1 = time.perf_counter()

# scipy_time = t1-t0

print(output1.shape)
print(output2.shape)

# print(output1)
# print(output2)
# print(correlate2d(input[0][0], kernel[0][0], mode="valid"))
# print(splatter_backward_filter(input[0][0], kernel[0][0]))

print(f"vanilla time: {vanilla_time}")
print(f"c optimized time: {c_time}")
# print(f"scipy time: {scipy_time}")
print(np.array_equal(output1, output2))



# for i in range(2, 25):
#     for k in range(2,25):

#         image = torch.randint(low=0, high=255, size=(i,i), dtype=torch.uint8)
#         kernel = torch.randint(low=0, high=5, size=(k,k))


#         image = image.numpy()

#         # print(kernel)
#         kernel = kernel.numpy()
#         image_base = convolve2d(image, kernel, mode= "valid")

#         image_splat = splat_conv2d(image, kernel, mode="valid")

#         # print(image_base)

#         # print()
#         # print(image_splat)

#         if(np.array_equal(image_base, image_splat)):
#             print("yay")
#         else:
#             print("boo", i, k)


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