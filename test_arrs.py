# import numpy as np
# import ctypes
# from numpy.ctypeslib import ndpointer 
# import time

# arr = np.array([
#     [1., 2., 3., 4.],
#     [5., 6., 7., 8.]
# ], dtype=np.float64)




# weightIndex = 1
# rows = 2
# cols = 4
# # arr = arr.reshape((1, 8))
# # print(input)
# # fun = ctypes.CDLL('C:\Users\trist\Documents\Dev\Data\libfun.so')
# lib = ctypes.CDLL('test_arrs.so')
# # print(nums)
# # fun.sting.argtypes =[ctypes.c_int32, ctypes.c_int32]


# _doublepp = ndpointer(dtype=np.float64, ndim=2) 
# lib.test.argtypes = (ctypes.c_int32, ctypes.c_int32, _doublepp)
# # lib.test.restype = ctypes.c_int32

# arrpp = (arr.__array_interface__['data'][0] 
#       + np.arange(arr.shape[0])*arr.strides[0]).astype(np.uintp) 

# lib.test(
#     rows,
#     cols,
#     arr
#     )

