from numpy import flip, float32
import numpy as np
# from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
# from splatterUpdate import splat_conv2d, splat_corr2d
import torch
# from torch.autograd.gradcheck import gradcheck
# from call_splatter import splatter_forward_full, splatter_backward_input_full, splatter_backward_filter_full
from scipy.signal import correlate2d, convolve2d
import splatter_cpp
import time



class Splatter_Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input.detach(), filter.detach(), bias.detach()
        input, filter = input.float(), filter.float()
        result = splatter_cpp.forward(input, filter)
        # result = splat_corr2d(input.numpy(), filter.numpy(), mode="valid") # supposed to be correlate
        result = bias + result
        ctx.save_for_backward(input, filter, bias)
        # return torch.as_tensor(result, dtype=input.dtype)
        # print(bias)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter, bias = ctx.saved_tensors
        grad_output = grad_output.numpy()
        grad_bias = np.sum(grad_output, keepdims=True)
        grad_input, grad_filter = splatter_cpp.backward(grad_output, filter, input)
        # the previous line can be expressed equivalently as:
        # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
        # grad_filter = splat_corr2d(input.numpy(), grad_output, mode="valid") # was correlate
        grad_filter = splatter_cpp.backward(input.numpy(), grad_output)
        return torch.from_numpy(grad_input), torch.from_numpy(grad_filter).to(torch.float), torch.from_numpy(grad_bias).to(torch.float)


class Splatter(Module):
    def __init__(self, filter_width, filter_height):
        super(Splatter, self).__init__()
        self.filter = Parameter(torch.randn(filter_width, filter_height))
        self.bias = Parameter(torch.randn(1, 1))

    def forward(self, input):
        return Splatter_Conv2d.apply(input, self.filter, self.bias)




"""testing  gradients show they are accuarate up to .001 which is not great"""

# torch.manual_seed(1)

input = torch.rand((1,1,8,8), dtype=torch.float32)
kernel = torch.rand((1,1,3,3), dtype=torch.float32)
input2 = torch.rand((1,1,10,10), dtype=torch.float32)

print(input)
print()

t0 = time.perf_counter()
output1 = splatter_cpp.backward(input, kernel, input2)[1]
t1 = time.perf_counter()

time1 = t1-t0

t0 = time.perf_counter()
output2 = correlate2d(input2[0][0].numpy(), input[0][0].numpy(), mode="valid")
t1 = time.perf_counter()

time2 = t1-t0



print(output1)
print()
print(torch.tensor(output2))


print(f"time 1: {time1}")
print(f"time 2: {time2}")


# # test 1
# module = Splatter(3, 3)
# # print("Filter and bias: ", list(module.parameters()))
# input = torch.randn(40, 1 , 10, 10, requires_grad=True)
# output = module(input)
# # print(output)
# print("Output from the convolution: ", output)
# # print("here?")
# # output.backward(torch.randn(8, 8))
# # print("Gradient for the input map: ", input.grad)

# print("test 1 passed")
# test 2


# moduleConv = Splatter(3, 3)

# input = [torch.randn(1, 1, 20, 20, dtype=torch.double, requires_grad=True)]
# test = gradcheck(moduleConv, input, eps=1e-3, atol=1e-4)
# print("Are the gradients correct: ", test)
