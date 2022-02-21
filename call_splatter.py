import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer 
from splatter import splat_corr2d, splat_conv2d
# import copy
# from scipy.signal import correlate2d, convolve2d




splatter_lib = ctypes.CDLL('splatterlib.so')
_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C') 
# splatter_lib.splatterForward.restype = ctypes.c_int32
splatter_lib.splatterForward.argtypes = (ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            _doublepp, _doublepp, _doublepp
            )
splatter_lib.splatterBackwardInput.argtypes = (ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            _doublepp, _doublepp, _doublepp
            )
splatter_lib.splatterBackwardFilter.argtypes = (ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            _doublepp, _doublepp, _doublepp
            )

def splatter_forward(input,kernel):
    
    weightIndex = kernel.shape[0] -2
    kernelSize = kernel.shape[0]
    rows, cols = input.shape
    padInput = np.zeros((rows + weightIndex*2, cols + weightIndex*2))
    padInput[weightIndex:-weightIndex, weightIndex:-weightIndex] = input
    output = np.zeros_like(padInput)

    inputpp = (padInput.__array_interface__['data'][0] 
        + np.arange(padInput.shape[0])*padInput.strides[0]).astype(np.uintp) 

    outputpp = (output.__array_interface__['data'][0] 
        + np.arange(output.shape[0])*output.strides[0]).astype(np.uintp) 

    kernelpp = (kernel.__array_interface__['data'][0] 
        + np.arange(kernel.shape[0])*kernel.strides[0]).astype(np.uintp) 

    splatter_lib.splatterForward(
        rows,
        cols,
        kernelSize,
        inputpp,
        outputpp,
        kernelpp
        )


    output = output[2*weightIndex:-2*weightIndex, 2*weightIndex:-2*weightIndex]
    return output

def splatter_forward_full(input, kernel):
    N , channels, height, width = input.shape
    weight_index = kernel.shape[0]-2
    input = np.array(input, dtype=np.float64)
    kernel = np.array(kernel, dtype=np.float64)

    output = np.zeros((N, channels, height-(2*weight_index), width-(2*weight_index)), dtype=np.float64)

    for img_index, image in enumerate(input):
        for chan_index, channel in enumerate(image):
            output[img_index][chan_index] = splatter_forward(channel ,  kernel)
    
    
    return output

def splatter_backward_input(input, kernel):
    
    weightIndex = int(kernel.shape[0]/2)
    kernelSize = kernel.shape[0]
    rows, cols = input.shape
    padInput = np.zeros((rows + weightIndex*2, cols + weightIndex*2))
    padInput[weightIndex:-weightIndex, weightIndex:-weightIndex] = input
    output = np.zeros_like(padInput)

    inputpp = (padInput.__array_interface__['data'][0] 
        + np.arange(padInput.shape[0])*padInput.strides[0]).astype(np.uintp) 

    outputpp = (output.__array_interface__['data'][0] 
        + np.arange(output.shape[0])*output.strides[0]).astype(np.uintp) 

    kernelpp = (kernel.__array_interface__['data'][0] 
        + np.arange(kernel.shape[0])*kernel.strides[0]).astype(np.uintp) 

    splatter_lib.splatterBackwardInput(
        rows,
        cols,
        kernelSize,
        inputpp,
        outputpp,
        kernelpp
        )


    # output = output[2*weightIndex:-2*weightIndex, 2*weightIndex:-2*weightIndex]
    return output

def splatter_backward_input_full(input, kernel):
    N , channels, height, width = input.shape
    weight_index = kernel.shape[0]-2
    input = np.array(input, dtype=np.float64)
    kernel = np.array(kernel, dtype=np.float64)

    output = np.zeros((N, channels, height+2, width+2), dtype=np.float64)

    for img_index, image in enumerate(input):
        for chan_index, channel in enumerate(image):
            output[img_index][chan_index] = splatter_backward_input(channel ,  kernel)
    
    return output

def splatter_backward_filter(input, kernel):
    print(input.shape,kernel.shape)
    weightIndex = int(kernel.shape[0]/2)
    kernelSize = kernel.shape[0]
    rows, cols = input.shape
    padInput = np.zeros((rows + weightIndex*2, cols + weightIndex*2))
    padInput[weightIndex:-weightIndex, weightIndex:-weightIndex] = input
    output = np.zeros_like(padInput)
    inputpp = (padInput.__array_interface__['data'][0] 
        + np.arange(padInput.shape[0])*padInput.strides[0]).astype(np.uintp) 

    outputpp = (output.__array_interface__['data'][0] 
        + np.arange(output.shape[0])*output.strides[0]).astype(np.uintp) 

    kernelpp = (kernel.__array_interface__['data'][0] 
        + np.arange(kernel.shape[0])*kernel.strides[0]).astype(np.uintp) 

    

    splatter_lib.splatterBackwardFilter(
        rows,
        cols,
        kernelSize,
        inputpp,
        outputpp,
        kernelpp
        )


    output =  output[2*weightIndex:-2*weightIndex +1,2*weightIndex:-2*weightIndex+1]
    # output =  output[2*weightIndex-1:-2*weightIndex ,2*weightIndex-1:-2*weightIndex]
    # output = output[2*weightIndex:-2*weightIndex, 2*weightIndex:-2*weightIndex]
    
    return output

def splatter_backward_filter_full(input, kernel):
    N , channels, height, width = input.shape
    weight_index = int(kernel.shape[2]/2)
    input = np.array(input, dtype=np.float64)
    kernel = np.array(kernel, dtype=np.float64)

    output = np.zeros((N, channels, abs(height-2*weight_index+1), abs(width-2*weight_index+1)), dtype=np.float64)
    # output = np.zeros((N, channels, height+8, width+8))

    for img_index, image in enumerate(input):
        for chan_index, channel in enumerate(image):
            output[img_index][chan_index] = splatter_backward_filter(input[img_index][chan_index] ,  kernel[img_index][chan_index])
    # output[:,:] = splatter_backward_filter(input[:,:] , kernel[:,:])
    # func = np.vectorize(splatter_backward_filter)
    # output[:,:] = splatter_backward_filter(input[:,0, None] , kernel[:,0, None])
    # np.fromfunction()
    return output

# input = np.random.rand(256)
# input = input.reshape((16,16))
# input2 = copy.deepcopy(input)
# kernel = np.random.rand(196)
# kernel = kernel.reshape((14,14))

# weightIndex = 1
# rows = 7
# cols = 7
# kernelSize = 3

# output1 = splatter_backward_filter(input, kernel)
# output2 = correlate2d(input, kernel, mode="valid")

# print(output1.shape)
# print(output2.shape)

# print(output1)
# print()
# print(output2)

# print(np.array_equal(output1, output2))

