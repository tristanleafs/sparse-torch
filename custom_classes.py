from numpy import flip
import numpy as np
# from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
# from splatterUpdate import splat_conv2d, splat_corr2d
import torch
# from torch.autograd.gradcheck import gradcheck
from call_splatter import splatter_forward_full, splatter_backward_input_full, splatter_backward_filter_full

import splatter_cpp



class Splatter_Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input.detach(), filter.detach(), bias.detach()
        # input, filter = input.float(), filter.float()
        
        # result = splatter_forward_full(input.numpy(), filter.numpy())
        result = splatter_cpp.forward(input, filter)

        # result = splat_corr2d(input.numpy(), filter.numpy(), mode="valid") # supposed to be correlate
        # result += bias.numpy()
        # torch.add(result, bias)
        ctx.save_for_backward(input, filter, bias)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter, bias = ctx.saved_tensors
        # grad_output = grad_output.numpy()
        grad_bias = np.sum(grad_output.numpy(), keepdims=True)

        # grad_bias = torch.sum(grad_output,dim=grad_output.dim, keepdim=True)

        # grad_input = splatter_backward_input_full(grad_output, filter.numpy())
        # the previous line can be expressed equivalently as:
        # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
        # grad_filter = splat_corr2d(input.numpy(), grad_output, mode="valid") # was correlate
        # grad_filter = splatter_backward_filter_full(input.numpy(), grad_output)

        grad_input = splatter_cpp.backward_input(grad_output, filter)
        grad_filter = splatter_cpp.backward_filter(input, grad_output)
        return grad_input, grad_filter, torch.tensor(grad_bias)


class Splatter(Module):
    def __init__(self, filter_width, filter_height):
        super(Splatter, self).__init__()
        self.filter = Parameter(torch.randn(filter_width, filter_height))
        self.bias = Parameter(torch.randn(1, 1))

    def forward(self, input):
        return Splatter_Conv2d.apply(input, self.filter, self.bias)

class Splatter_Conv2d_non_sparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input.detach(), filter.detach(), bias.detach()
        # input, filter = input.float(), filter.float()
        
        # result = splatter_forward_full(input.numpy(), filter.numpy())
        result = splatter_cpp.forward_non_sparse(input, filter)

        # result = splat_corr2d(input.numpy(), filter.numpy(), mode="valid") # supposed to be correlate
        # result += bias.numpy()
        # torch.add(result, bias)
        ctx.save_for_backward(input, filter, bias)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter, bias = ctx.saved_tensors
        # grad_output = grad_output.numpy()
        grad_bias = np.sum(grad_output.numpy(), keepdims=True)

        # grad_bias = torch.sum(grad_output,dim=grad_output.dim, keepdim=True)

        # grad_input = splatter_backward_input_full(grad_output, filter.numpy())
        # the previous line can be expressed equivalently as:
        # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
        # grad_filter = splat_corr2d(input.numpy(), grad_output, mode="valid") # was correlate
        # grad_filter = splatter_backward_filter_full(input.numpy(), grad_output)

        grad_input = splatter_cpp.backward_input(grad_output, filter)
        grad_filter = splatter_cpp.backward_filter(input, grad_output)
        return grad_input, grad_filter, torch.tensor(grad_bias)


class Splatter_Non_Sparse(Module):
    def __init__(self, filter_width, filter_height):
        super(Splatter_Non_Sparse, self).__init__()
        self.filter = Parameter(torch.randn(filter_width, filter_height))
        self.bias = Parameter(torch.randn(1, 1))

    def forward(self, input):
        return Splatter_Conv2d_non_sparse.apply(input, self.filter, self.bias)

def get_sparcity(tensor):
    return float(1-(tensor.count_nonzero().item())/float(tensor.numel()))

class Save_Output_body(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, filter, file_name):
    
        f = open(file_name, 'a')
        f.write(str(get_sparcity(input)))
        f.write(',')
        f.close()
        ctx.save_for_backward(input)
        # torch.add(result, bias)
        return input

    @staticmethod
    def backward(ctx, output):
        
        return output


class Save_Output(Module):
    def __init__(self, file_name: str):
        super(Save_Output, self).__init__()
        self.bias = Parameter(torch.randn(1,1))
        self.filter = Parameter(torch.randn(1,1))
        self.file_name = file_name
        
    def forward(self, input):
        return Save_Output_body.apply(input, self.bias, self.filter, self.file_name )




"""testing  gradients show they are accuarate up to .001 which is not great"""

# # # test 1
# module = Splatter(3, 3)
# print("Filter and bias: ", list(module.parameters()))
# input = torch.randn(40, 1,10, 10, requires_grad=True)
# output = module(input)
# print("Output from the convolution: ", output)
# print("here?")
# output.backward(torch.randn(8, 8))
# print("Gradient for the input map: ", input.grad)

# print("test 1 passed")
# test 2

# from torch.autograd.gradcheck import gradcheck

# moduleConv = Save_Output()

# input = [torch.randn(1, 1, 20, 20, dtype=torch.double, requires_grad=True)]
# test = gradcheck(moduleConv, input, eps=1e-3, atol=1e-4)
# print("Are the gradients correct: ", test)
