from numpy import flip
import numpy as np
from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

import torch
from torch.autograd.gradcheck import gradcheck


def splat_conv2d(input, kernel):
    #this function skips reversing the kernel
    #establish size of input image
    
    rows = input.shape[0]
    cols = input.shape[1]

    #establish weight kernel size
    weightSize = kernel.shape[0]

    #establish weight indexing
    weightIndex = int(weightSize/2) 

    #pad input matrix
    padInput = np.zeros((rows + weightIndex*2, cols + weightIndex*2))
    padInput[weightIndex:-weightIndex, weightIndex:-weightIndex] = input

    #create padded output image
    output = np.zeros_like(padInput)


    #iterate though input matrix
    for i in range(weightIndex, weightIndex+rows): #iterate through rows of input
        for j in range(weightIndex, weightIndex+cols): #iterate through columns of input
            
            #check if index is nonzero
            if(padInput[i][j] > 0 or padInput[i][j] <0):

                #update output using vector operations
                output[i-weightIndex:i+weightIndex +1,j-weightIndex:j+weightIndex+1] += padInput[i][j]*kernel


    #adjust final output size using splicing
    output = output[weightIndex:-weightIndex, weightIndex:-weightIndex]
    return output
    #end of function


class Splatter_Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input.detach(), filter.detach(), bias.detach()
        result = splat_conv2d(input.numpy(), filter.numpy())
        result += bias.numpy()
        ctx.save_for_backward(input, filter, bias)
        return torch.as_tensor(result, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter, bias = ctx.saved_tensors
        grad_output = grad_output.numpy()
        grad_bias = np.sum(grad_output, keepdims=True)
        grad_input = splat_conv2d(grad_output, filter.numpy(), mode='full')
        # the previous line can be expressed equivalently as:
        # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
        grad_filter = correlate2d(input.numpy(), grad_output, mode='valid')
        return torch.from_numpy(grad_input), torch.from_numpy(grad_filter).to(torch.float), torch.from_numpy(grad_bias).to(torch.float)


class ScipyConv2d(Module):
    def __init__(self, filter_width, filter_height):
        super(ScipyConv2d, self).__init__()
        self.filter = Parameter(torch.randn(filter_width, filter_height))
        self.bias = Parameter(torch.randn(1, 1))

    def forward(self, input):
        return Splatter_Conv2d.apply(input, self.filter, self.bias)

# test 1
module = ScipyConv2d(3, 3)
print("Filter and bias: ", list(module.parameters()))
input = torch.randn(10, 10, requires_grad=True)
output = module(input)
print("Output from the convolution: ", output)
print("here?")
output.backward(torch.randn(8, 8))
print("Gradient for the input map: ", input.grad)

print("test 1 passed")
# test 2


moduleConv = ScipyConv2d(3, 3)

input = [torch.randn(20, 20, dtype=torch.double, requires_grad=True)]
test = gradcheck(moduleConv, input, eps=1e-6, atol=1e-4)
print("Are the gradients correct: ", test)
