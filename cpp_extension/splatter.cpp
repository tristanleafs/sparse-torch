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

  int rowsPadded = rows + 2*kernelIndex;
  int colsPadded = cols + 2*kernelIndex;

  int rowsFinal = rows - 2*kernelIndex;
  int colsFinal = cols - 2*kernelIndex;


  torch::Tensor output = torch::zeros({batch, channels, rows-(2*kernelIndex), cols-(2*kernelIndex)});
  torch::Tensor outputPadded = torch::zeros({batch, channels, rows+(2*kernelIndex), cols+(2*kernelIndex)});

  float* input_arr = (float*)input.data_ptr();
  float* kernel_arr = (float*)kernel.data_ptr();
  float* output_arr = (float*)output.data_ptr();
  float* output_padded_arr = (float*)outputPadded.data_ptr();

  for(int imageIndex = 0; imageIndex< batch; imageIndex++)
  {
    for(int channel = 0; channel < channels; channel++)
    {
      for(int row = kernelIndex; row < rows + kernelIndex; row++)
      {
        for(int col = kernelIndex; col < cols + kernelIndex; col++)
        {
          float inputTemp = *(input_arr +(col -kernelIndex + (row-kernelIndex)*cols + channel*rows*cols + imageIndex*channels*rows*cols));
          // std::cout<<inputTemp << " ";
          if(inputTemp > .00001)
          {
            //kernel multiplication
            for(int k = -1*kernelIndex; k<= kernelIndex; k++)
            {
              for(int l = -1*kernelIndex; l <= kernelIndex; l++)
              {
                
                
                // std::cout << *(output_arr+(col+l + (row+k)*cols + channel*rows*cols + imageIndex*channel*rows*cols));

                *(output_padded_arr+(col-l + (row-k )*colsPadded + channel*rowsPadded*colsPadded + imageIndex*channels*rowsPadded*colsPadded)) 
                += *(kernel_arr+(l+kernelIndex + (k+kernelIndex)*kernelSize)) * inputTemp;
                
                // std::cout<< row-k << " " << col-l << " "<< *(output_padded_arr+(col-l + (row-k )*cols + channel*rows*cols + imageIndex*channels*rows*cols)) <<std::endl;
                
              }
            }
          }

        }
      }
      for(int row = 0; row < rows -2*kernelIndex; row++)
      {
        for(int col = 0; col < cols - 2*kernelIndex; col++)
        {
          *(output_arr + col + row*colsFinal + channel*rowsFinal*colsFinal + imageIndex*channels*rowsFinal*colsFinal) 
          = *(output_padded_arr + col + 2*kernelIndex + (row+2*kernelIndex)*colsPadded + channel*rowsPadded*colsPadded + imageIndex*channels*rowsPadded*colsPadded);
        }
        
      }
      //end of channel
    }
  }

  // std::cout << std::endl << *output_arr << " " << *(output_arr+1) << std::endl;

  

  // for(int row = 0; row < rows+2; row++)
  // {
  //   for(int col = 0; col < cols+2; col++)
  //   {
  //     std::cout << *(output_padded_arr + col + row*colsPadded) << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // for(int row = 0; row < rows-2; row++)
  // {
  //   for(int col = 0; col < cols-2; col++)
  //   {
  //     std::cout << *(output_arr + col + row*colsFinal) << " ";
  //   }
  //   std::cout << std::endl << std::endl;
  // }

  auto options = torch::TensorOptions().dtype(torch::kFloat32);

  //{batch*channels*(rows)*(cols),(rows)*(cols),cols,1}

  output = torch::from_blob(output_arr, {batch, channels, rowsFinal, colsFinal}, options);
  // delete output_arr;
  // delete output_padded_arr;
  // delete kernel_arr;
  // delete input_arr;
  return output.clone();
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