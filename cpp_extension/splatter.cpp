#include <torch/extension.h>
#include <vector>
#include <stdio.h>

// s'(z) = (1 - s(z)) * s(z)
// torch::Tensor d_sigmoid(torch::Tensor z) {
//   auto s = torch::sigmoid(z);
//   return (1 - s) * s;
// }

// // tanh'(z) = 1 - tanh^2(z)
// torch::Tensor d_tanh(torch::Tensor z) {
//   return 1 - z.tanh().pow(2);
// }

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
// torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
//   auto e = z.exp();
//   auto mask = (alpha * (e - 1)) < 0;
//   return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
// }



torch::Tensor splatter_forward(
    torch::Tensor input,
    torch::Tensor kernel) {
  
  int batch = input.size(0);
  int channels = input.size(1);
  int rows = input.size(2);
  int cols = input.size(3);

  int kernelSize = kernel.size(0);
  int kernelIndex = kernelSize/2;

  torch::Tensor output = torch::zeros({batch, channels, rows-(2*kernelIndex), cols-(2*kernelIndex)});

  for(int imageIndex = 0; imageIndex< batch; imageIndex++)
  {
    for(int channel = 0; channel < channels; channel++)
    {
      //padded output for more stuff
      auto outputPadded = torch::zeros({rows+(2*kernelIndex), cols+(2*kernelIndex)});

      for(int row = kernelIndex; row < rows + kernelIndex; row++ )
      {
        for(int col = kernelIndex; col < kernelIndex + cols; col++)
        {
          //kernel multiplication
          auto inputTemp = input[imageIndex][channel][row-kernelIndex][col-kernelIndex];
          if(true)
          {

          
            // for(int k = -1*kernelIndex; k <= kernelIndex; k++)
            // {
            //   for(int l = -1*kernelIndex; l<=kernelIndex; l++)
            //   {
            //     outputPadded[row-k][col-l] += inputTemp*kernel[kernelIndex+k][kernelIndex+l];
            //   }
            // }
            
          }
        }
      }
      
      for(int row = 0; row < rows-(2*kernelIndex); row++)
      {
        for(int col = 0; col < cols-(2*kernelIndex) ;col++){
          output[imageIndex][channel][row][col] = outputPadded[row+(kernelIndex*2)][col+(kernelIndex*2)];
        }
      }
      // output[imageIndex][channel] = torch::ones({rows-(2*kernelIndex), cols-(2*kernelIndex)});
    }
  }

  return output;
}

std::vector<torch::Tensor> splatter_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell
    // torch::Tensor new_cell,
    // torch::Tensor input_gate,
    // torch::Tensor output_gate,
    // torch::Tensor candidate_cell,
    // torch::Tensor X,
    // torch::Tensor gate_weights,
    // torch::Tensor weights
    ) {
    
    std::cout<<"backward";

  return {grad_cell};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &splatter_forward, "splatter forward");
  m.def("backward", &splatter_backward, "splatter backward");
}