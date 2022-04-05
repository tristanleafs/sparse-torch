
from scipy.signal import convolve2d, correlate2d
import torch
import numpy as np
import splatter_cpp

output = torch.rand((20,1,8,8), dtype=torch.float32)
kernel = torch.rand((3,3), dtype=torch.float32)
input = torch.rand((20,1,200,200), dtype=torch.float32)

def forward_test():
    splatter_output = splatter_cpp.forward_non_sparse(input, kernel)
    expected_output = correlate2d(input[0][0], kernel, mode='valid')
    expected_output2 = correlate2d(input[19][0], kernel, mode='valid')
    if(np.array_equal(splatter_output[0][0] , expected_output)):
        pass
    else:
        print(splatter_output[0][0])
        print(expected_output)
    assert np.array_equal(splatter_output[19][0], expected_output2)
    print("we made it")

forward_test()