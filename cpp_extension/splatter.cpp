#include <torch/extension.h>
#include <vector>
#include <stdio.h>
#include <chrono>

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
  

  // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  int batch = input.size(0);
  int channels = input.size(1);
  int rows = input.size(2);
  int cols = input.size(3);
  

  int kernelSize = kernel.size(0);
  int kernelIndex = kernelSize/2;
  int largeIndex = 2*kernelIndex;
  int rowsPadded = rows + 2*kernelIndex;
  int colsPadded = cols + 2*kernelIndex;

  int rowsFinal = rows - largeIndex;
  int colsFinal = cols - largeIndex;


  // torch::Tensor output = torch::zeros({batch, channels, rows-(2*kernelIndex), cols-(2*kernelIndex)});
  // torch::Tensor outputPadded = torch::zeros({batch, channels, rows+(2*kernelIndex), cols+(2*kernelIndex)});

  int outputSize = batch*channels*rowsFinal*colsFinal;
  int outputPaddedSize = batch*channels*rowsPadded*colsPadded;

  // float* input_arr{new float[batch*channels*rows*cols]{input.data_ptr<float>()}};
  // float* kernel_arr{new float(kernelSize*kernelSize){(float*)kernel.data_ptr()}};
  // float* output_arr {new float(batch*channels*rowsFinal*colsFinal){(float*)output.data_ptr()}};
  // float* output_padded_arr{ new float(batch*channels*rowsPadded*colsPadded){(float*)outputPadded.data_ptr()}};


  float* input_arr = (float*)input.data_ptr();
  float* kernel_arr = (float*)kernel.data_ptr();
  // float* output_arr = (float*)output.data_ptr();
  // float* output_padded_arr = (float*)outputPadded.data_ptr();

  float* output_arr = new float[outputSize]();
  float* output_padded_arr = new float[outputPaddedSize]();

  // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
  // float* output_arr = &output_arr;
  // float* output_padded_arr = &output_padded_arr1;
  // begin = std::chrono::steady_clock::now();
  
  for(int imageIndex = 0; imageIndex< batch; imageIndex++)
  {
    
    for(int channel = 0; channel < channels; channel++)
    {
      int padOffset = channel*rowsPadded*colsPadded + imageIndex*channels*rowsPadded*colsPadded;
      int finalOffset = channel*rowsFinal*colsFinal + imageIndex*channels*rowsFinal*colsFinal;
      // std::cout << padOffset << std::endl;
      for(int row = kernelIndex; row < rows + kernelIndex; row++)
      {
        for(int col = kernelIndex; col < cols + kernelIndex; col++)
        {
          // float inputTemp = *(input_arr +(col -kernelIndex + (row-kernelIndex)*cols + channel*rows*cols + imageIndex*channels*rows*cols));
          float inputTemp = *input_arr;
          input_arr++;
          // std::cout<<inputTemp << " ";
          if(inputTemp > .00002)
          {
            //kernel multiplication
            // int padOffset = channel*rowsPadded*colsPadded + imageIndex*channels*rowsPadded*colsPadded;
            for(int k = -1*kernelIndex; k<= kernelIndex; k++)
            {
              for(int l = -1*kernelIndex; l <= kernelIndex; l++)
              {
                
                
                // std::cout << *(output_arr+(col+l + (row+k)*cols + channel*rows*cols + imageIndex*channel*rows*cols));
                // if(row-k-largeIndex >= 0 && col-l -largeIndex >= 0 && row-k  -largeIndex < rowsFinal  && col-l -largeIndex < colsFinal){
                *(output_padded_arr+(col-l + (row-k )*colsPadded + padOffset)) 
                += *(kernel_arr+(l+kernelIndex + (k+kernelIndex)*kernelSize)) * inputTemp;
                // }
                
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
          *(output_arr + col + row*colsFinal + finalOffset) 
          = *(output_padded_arr + col + 2*kernelIndex + (row+2*kernelIndex)*colsPadded + padOffset);
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

  // auto options = torch::TensorOptions().dtype(torch::kFloat32);

  //{batch*channels*(rows)*(cols),(rows)*(cols),cols,1}
  // end = std::chrono::steady_clock::now();
  // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
  // begin = std::chrono::steady_clock::now();
  torch::Tensor new_output = torch::from_blob(output_arr, {batch, channels, rowsFinal, colsFinal});
  // delete output_arr;
  // delete output_padded_arr;
  // delete kernel_arr;
  // delete input_arr;
  

  new_output = new_output.clone();
  delete [] output_arr;
  delete [] output_padded_arr;
  // end = std::chrono::steady_clock::now();
  // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
  return (new_output);
}


torch::Tensor splatter_forward_non_sparse(
    torch::Tensor input,
    torch::Tensor kernel) {
  

  // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  int batch = input.size(0);
  int channels = input.size(1);
  int rows = input.size(2);
  int cols = input.size(3);
  

  int kernelSize = kernel.size(0);
  int kernelIndex = kernelSize/2;
  int largeIndex = 2*kernelIndex;
  int rowsPadded = rows + 2*kernelIndex;
  int colsPadded = cols + 2*kernelIndex;

  int rowsFinal = rows - largeIndex;
  int colsFinal = cols - largeIndex;


  // torch::Tensor output = torch::zeros({batch, channels, rows-(2*kernelIndex), cols-(2*kernelIndex)});
  // torch::Tensor outputPadded = torch::zeros({batch, channels, rows+(2*kernelIndex), cols+(2*kernelIndex)});

  int outputSize = batch*channels*rowsFinal*colsFinal;
  int outputPaddedSize = batch*channels*rowsPadded*colsPadded;

  // float* input_arr{new float[batch*channels*rows*cols]{input.data_ptr<float>()}};
  // float* kernel_arr{new float(kernelSize*kernelSize){(float*)kernel.data_ptr()}};
  // float* output_arr {new float(batch*channels*rowsFinal*colsFinal){(float*)output.data_ptr()}};
  // float* output_padded_arr{ new float(batch*channels*rowsPadded*colsPadded){(float*)outputPadded.data_ptr()}};


  float* input_arr = (float*)input.data_ptr();
  float* kernel_arr = (float*)kernel.data_ptr();
  // float* output_arr = (float*)output.data_ptr();
  // float* output_padded_arr = (float*)outputPadded.data_ptr();

  float* output_arr = new float[outputSize]();
  float* output_padded_arr = new float[outputPaddedSize]();

  // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
  // float* output_arr = &output_arr;
  // float* output_padded_arr = &output_padded_arr1;
  // begin = std::chrono::steady_clock::now();
  
  for(int imageIndex = 0; imageIndex< batch; imageIndex++)
  {
    
    for(int channel = 0; channel < channels; channel++)
    {
      int padOffset = channel*rowsPadded*colsPadded + imageIndex*channels*rowsPadded*colsPadded;
      int finalOffset = channel*rowsFinal*colsFinal + imageIndex*channels*rowsFinal*colsFinal;
      // std::cout << padOffset << std::endl;
      for(int row = kernelIndex; row < rows + kernelIndex; row++)
      {
        for(int col = kernelIndex; col < cols + kernelIndex; col++)
        {
          // float inputTemp = *(input_arr +(col -kernelIndex + (row-kernelIndex)*cols + channel*rows*cols + imageIndex*channels*rows*cols));
          float inputTemp = *input_arr;
          input_arr++;
          // std::cout<<inputTemp << " ";
          // if(inputTemp > .00002)
          // {
            //kernel multiplication
            // int padOffset = channel*rowsPadded*colsPadded + imageIndex*channels*rowsPadded*colsPadded;
            for(int k = -1*kernelIndex; k<= kernelIndex; k++)
            {
              for(int l = -1*kernelIndex; l <= kernelIndex; l++)
              {
                
                
                // std::cout << *(output_arr+(col+l + (row+k)*cols + channel*rows*cols + imageIndex*channel*rows*cols));
                // if(row-k-largeIndex >= 0 && col-l -largeIndex >= 0 && row-k  -largeIndex < rowsFinal  && col-l -largeIndex < colsFinal){
                *(output_padded_arr+(col-l + (row-k )*colsPadded + padOffset)) 
                += *(kernel_arr+(l+kernelIndex + (k+kernelIndex)*kernelSize)) * inputTemp;
                // }
                
                // std::cout<< row-k << " " << col-l << " "<< *(output_padded_arr+(col-l + (row-k )*cols + channel*rows*cols + imageIndex*channels*rows*cols)) <<std::endl;
                
              }
            }
          // }

        }
      }
      
      
      for(int row = 0; row < rows -2*kernelIndex; row++)
      {
        for(int col = 0; col < cols - 2*kernelIndex; col++)
        {
          *(output_arr + col + row*colsFinal + finalOffset) 
          = *(output_padded_arr + col + 2*kernelIndex + (row+2*kernelIndex)*colsPadded + padOffset);
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

  // auto options = torch::TensorOptions().dtype(torch::kFloat32);

  //{batch*channels*(rows)*(cols),(rows)*(cols),cols,1}
  // end = std::chrono::steady_clock::now();
  // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
  // begin = std::chrono::steady_clock::now();
  torch::Tensor new_output = torch::from_blob(output_arr, {batch, channels, rowsFinal, colsFinal});
  // delete output_arr;
  // delete output_padded_arr;
  // delete kernel_arr;
  // delete input_arr;
  

  new_output = new_output.clone();
  delete [] output_arr;
  delete [] output_padded_arr;
  // end = std::chrono::steady_clock::now();
  // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
  return (new_output);
}

std::vector<torch::Tensor> splatter_backward(
    torch::Tensor grad_output,
    torch::Tensor kernel,
    torch::Tensor input
    // torch::Tensor new_cell,
    // torch::Tensor input_gate,
    // torch::Tensor output_gate,
    // torch::Tensor candidate_cell,
    // torch::Tensor X,
    // torch::Tensor gate_weights,
    // torch::Tensor weights
    ) 
{
      
  int batch = input.size(0);
  int channels = input.size(1);
  int input_rows = input.size(2);
  int input_cols = input.size(3);

  int grad_output_rows = grad_output.size(2);
  int grad_output_cols = grad_output.size(3);
  

  int kernelSize = kernel.size(0);
  int kernelIndex = kernelSize/2;

  int grad_input_rows = grad_output_rows + 2*kernelIndex;
  int grad_input_cols = grad_output_cols + 2*kernelIndex;

  int grad_index_y = grad_output_rows/2;
  int grad_index_x = grad_output_cols/2;

  int rowsPadded = input_rows + 2*grad_index_y;
  int colsPadded = input_cols + 2*grad_index_x;

  int rowsFinal = input_rows - 2*grad_index_y+1;
  int colsFinal = input_cols - 2*grad_index_x+1;




  torch::Tensor output = torch::zeros({batch, channels, rowsFinal, colsFinal});
  torch::Tensor outputPadded = torch::zeros({batch, channels, rowsPadded, colsPadded});
  torch::Tensor gradPadded = torch::zeros({batch, channels, grad_input_rows, grad_input_cols});

  // float* input_arr{new float[batch*channels*rows*cols]{input.data_ptr<float>()}};
  // float* kernel_arr{new float(kernelSize*kernelSize){(float*)kernel.data_ptr()}};
  // float* output_arr {new float(batch*channels*rowsFinal*colsFinal){(float*)output.data_ptr()}};
  // float* output_padded_arr{ new float(batch*channels*rowsPadded*colsPadded){(float*)outputPadded.data_ptr()}};


  float* input_arr = (float*)input.data_ptr();
  float* kernel_arr = (float*)kernel.data_ptr();
  float* output_arr = (float*)output.data_ptr();
  float* grad_output_arr = (float*)grad_output.data_ptr();
  float* output_padded_arr = (float*)outputPadded.data_ptr();
  float* grad_input_arr = (float*)gradPadded.data_ptr();

  //find grad_input
  for(int imageIndex = 0; imageIndex< batch; imageIndex++)
  {
    for(int channel = 0; channel < channels; channel++)
    {
      for(int row = kernelIndex; row < grad_output_rows + kernelIndex; row++)
      {
        for(int col = kernelIndex; col < grad_output_cols + kernelIndex; col++)
        {
          float gradTemp = *(grad_output_arr +(col -kernelIndex + (row-kernelIndex)*grad_output_cols 
          + channel*grad_output_rows*grad_output_cols + imageIndex*channels*grad_output_rows*grad_output_cols));
          // std::cout<<inputTemp << " ";
          if(gradTemp > .00001)
          {
            //kernel multiplication
            for(int k = -1*kernelIndex; k<= kernelIndex; k++)
            {
              for(int l = -1*kernelIndex; l <= kernelIndex; l++)
              {
                
                
                // std::cout << *(output_arr+(col+l + (row+k)*cols + channel*rows*cols + imageIndex*channel*rows*cols));

                *(grad_input_arr+(col+l + (row+k )*grad_input_cols + channel*grad_input_rows*grad_input_cols 
                + imageIndex*channels*grad_input_rows*grad_input_cols)) 
                += *(kernel_arr+(l+kernelIndex + (k+kernelIndex)*kernelSize)) 
                * gradTemp;
                
                // std::cout<< row-k << " " << col-l << " "<< *(output_padded_arr+(col-l + (row-k )*cols + channel*rows*cols + imageIndex*channels*rows*cols)) <<std::endl;
                
              }
            }
          }

        }
      }

     
    }
  }

  //calculate filter thingy
  for(int imageIndex = 0; imageIndex< batch; imageIndex++)
  {
    for(int channel = 0; channel < channels; channel++)
    {
      for(int row = grad_index_y; row < input_rows + grad_index_y; row++)
      {
        for(int col = grad_index_x; col < input_cols + grad_index_x; col++)
        {
          float inputTemp = *(input_arr +(col -grad_index_x + (row-grad_index_y)*input_cols + channel*input_rows*input_cols + imageIndex*channels*input_rows*input_cols));
          // std::cout<<inputTemp << " ";
          if(inputTemp > .00001)
          {
            //kernel multiplication
            for(int k = -1*grad_index_y; k< grad_index_y; k++)
            {
              for(int l = -1*grad_index_x; l < grad_index_x; l++)
              {
                
                
                // std::cout << *(output_arr+(col+l + (row+k)*cols + channel*rows*cols + imageIndex*channel*rows*cols));

                *(output_padded_arr+(col-l + (row-k )*colsPadded + channel*rowsPadded*colsPadded + imageIndex*channels*rowsPadded*colsPadded)) 
                += *(grad_output_arr+(l+grad_index_x + (k+grad_index_x)*grad_output_cols + channel*grad_output_rows*grad_output_cols
                 + imageIndex*channels*grad_output_rows*grad_output_cols)) 
                * inputTemp;
                
                // std::cout<< row-k << " " << col-l << " "<< *(output_padded_arr+(col-l + (row-k )*cols + channel*rows*cols + imageIndex*channels*rows*cols)) <<std::endl;
                
              }
            }
          }

        }
      }
      
      for(int row = 0; row < input_rows -2*grad_index_y+1; row++)
      {
        for(int col = 0; col < input_cols - 2*grad_index_x +1; col++)
        {
          *(output_arr + col + row*colsFinal + channel*rowsFinal*colsFinal + imageIndex*channels*rowsFinal*colsFinal) 
          = *(output_padded_arr + col + 2*grad_index_x + (row+2*grad_index_y)*colsPadded + channel*rowsPadded*colsPadded + imageIndex*channels*rowsPadded*colsPadded);
        }
        
      }
      // end of channel
      // for(int row = 0; row < rowsPadded; row++)
      // {
      //   for(int col = 0; col < colsPadded; col++)
      //   {
      //     std::cout << *(output_padded_arr + col + row*colsPadded + channel*rowsFinal*colsFinal + imageIndex*channels*rowsFinal*colsFinal) << " ";
          
      //   }
      //   std::cout << std::endl;
        
      // }

    }
  }

  

  // auto options = torch::TensorOptions().dtype(torch::kFloat32);

  //{batch*channels*(rows)*(cols),(rows)*(cols),cols,1}
  
  torch::Tensor new_grad_input = torch::from_blob(grad_input_arr, {batch, channels, grad_input_rows, grad_input_cols});
  
  torch::Tensor new_output = torch::from_blob(output_arr, {batch, channels, rowsFinal, colsFinal});
  // std::cout << "here " << batch << new_output << std::endl;
  // delete output_arr;
  // delete output_padded_arr;
  // delete kernel_arr;
  // delete input_arr;
  return {new_grad_input.clone(), new_output.clone()}; 
  


  
}

torch::Tensor splatter_backward_input(
    torch::Tensor grad_output,
    torch::Tensor kernel
    ) 
{
      
  int batch = grad_output.size(0);
  int channels = grad_output.size(1);

  int grad_output_rows = grad_output.size(2);
  int grad_output_cols = grad_output.size(3);
  

  int kernelSize = kernel.size(0);
  int kernelIndex = kernelSize/2;

  int grad_input_rows = grad_output_rows + 2*kernelIndex;
  int grad_input_cols = grad_output_cols + 2*kernelIndex;
  

  torch::Tensor gradPadded = torch::zeros({batch, channels, grad_input_rows, grad_input_cols});

  // float* input_arr{new float[batch*channels*rows*cols]{input.data_ptr<float>()}};
  // float* kernel_arr{new float(kernelSize*kernelSize){(float*)kernel.data_ptr()}};
  // float* output_arr {new float(batch*channels*rowsFinal*colsFinal){(float*)output.data_ptr()}};
  // float* output_padded_arr{ new float(batch*channels*rowsPadded*colsPadded){(float*)outputPadded.data_ptr()}};


  float* kernel_arr = (float*)kernel.data_ptr();
  float* grad_input_arr = (float*)gradPadded.data_ptr();
  float* grad_output_arr = (float*)grad_output.data_ptr();

  //find grad_input
  for(int imageIndex = 0; imageIndex< batch; imageIndex++)
  {
    for(int channel = 0; channel < channels; channel++)
    {
      for(int row = kernelIndex; row < grad_output_rows + kernelIndex; row++)
      {
        for(int col = kernelIndex; col < grad_output_cols + kernelIndex; col++)
        {
          float gradTemp = *(grad_output_arr +(col -kernelIndex + (row-kernelIndex)*grad_output_cols 
          + channel*grad_output_rows*grad_output_cols + imageIndex*channels*grad_output_rows*grad_output_cols));
          // std::cout<<inputTemp << " ";
          if(gradTemp > .00001)
          {
            //kernel multiplication
            for(int k = -1*kernelIndex; k<= kernelIndex; k++)
            {
              for(int l = -1*kernelIndex; l <= kernelIndex; l++)
              {
                
                
                // std::cout << *(output_arr+(col+l + (row+k)*cols + channel*rows*cols + imageIndex*channel*rows*cols));

                *(grad_input_arr+(col+l + (row+k )*grad_input_cols + channel*grad_input_rows*grad_input_cols 
                + imageIndex*channels*grad_input_rows*grad_input_cols)) 
                += *(kernel_arr+(l+kernelIndex + (k+kernelIndex)*kernelSize)) 
                * gradTemp;
                
                // std::cout<< row-k << " " << col-l << " "<< *(output_padded_arr+(col-l + (row-k )*cols + channel*rows*cols + imageIndex*channels*rows*cols)) <<std::endl;
                
              }
            }
          }

        }
      }

     
    }
  }
  
  // auto options = torch::TensorOptions().dtype(torch::kFloat32);

  //{batch*channels*(rows)*(cols),(rows)*(cols),cols,1}
  
  torch::Tensor new_grad_input = torch::from_blob(grad_input_arr, {batch, channels, grad_input_rows, grad_input_cols});
  
  return new_grad_input.clone();
  
}

torch::Tensor splatter_backward_filter(
    
    torch::Tensor input,
    torch::Tensor grad_output
    
    ) 
{
      
  int batch = input.size(0);
  int channels = input.size(1);
  int input_rows = input.size(2);
  int input_cols = input.size(3);

  int grad_output_rows = grad_output.size(2);
  int grad_output_cols = grad_output.size(3);
  
  int grad_index_y = grad_output_rows/2;
  int grad_index_x = grad_output_cols/2;

  // int rowsPadded = input_rows + 2*grad_index_y;
  // int colsPadded = input_cols + 2*grad_index_x;

  int rowsFinal = input_rows - 2*grad_index_y+1;
  int colsFinal = input_cols - 2*grad_index_x+1;

 


  torch::Tensor output = torch::zeros({batch, channels, rowsFinal, colsFinal});
  // torch::Tensor outputPadded = torch::zeros({batch, channels, rowsPadded, colsPadded});

  // float* input_arr{new float[batch*channels*rows*cols]{input.data_ptr<float>()}};
  // float* kernel_arr{new float(kernelSize*kernelSize){(float*)kernel.data_ptr()}};
  // float* output_arr {new float(batch*channels*rowsFinal*colsFinal){(float*)output.data_ptr()}};
  // float* output_padded_arr{ new float(batch*channels*rowsPadded*colsPadded){(float*)outputPadded.data_ptr()}};


  float* input_arr = (float*)input.data_ptr();
  float* output_arr = (float*)output.data_ptr();
  float* grad_output_arr = (float*)grad_output.data_ptr();
  // float* output_padded_arr = (float*)outputPadded.data_ptr();

  // float output_arr1[batch][channels][rowsFinal][colsFinal] = {0};
  // float output_padded_arr1[batch][channels][rowsPadded][colsPadded] = {0};

  // float* output_arr = new float[batch*channels*rowsFinal*colsFinal] ;
  // float* output_padded_arr = new float[batch*channels*rowsPadded*colsPadded];


  //calculate filter thingy
  for(int imageIndex = 0; imageIndex< batch; imageIndex++)
  {
    for(int channel = 0; channel < channels; channel++)
    {
      for(int row = 0; row < rowsFinal; row++)
      {
        for(int col = 0; col < colsFinal ; col++)
        {
          int outputTempIndex =  (col + (row)*colsFinal + channel*colsFinal*rowsFinal + imageIndex*channels*rowsFinal*colsFinal);
          
          // std::cout<<outputTempIndex << std::endl;
          
          //kernel multiplication
          for(int k = -1*grad_index_y; k< grad_index_y; k++)
          {
            for(int l = -1*grad_index_x; l < grad_index_x; l++)
            {
              
              
              // std::cout << *(output_arr+(col+l + (row+k)*cols + channel*rows*cols + imageIndex*channel*rows*cols));
              float inputTemp = *(input_arr+(l+grad_index_x+col + (k+grad_index_x+row)*input_cols + channel*input_rows*input_cols
                + imageIndex*channels*input_rows*input_cols));
              *(output_arr + col + row*colsFinal + channel*colsFinal*rowsFinal + imageIndex*channels*rowsFinal*colsFinal)
              += *(grad_output_arr + (l+grad_index_x + (k+grad_index_x)*grad_output_cols + channel*grad_output_rows*grad_output_cols 
              + imageIndex*channels*grad_output_rows*grad_output_cols)) * inputTemp;
              
              // std::cout<< *(output_arr + outputTempIndex) << " " << inputTemp << " " << *(grad_output_arr + (l+grad_index_x + (k+grad_index_x)*grad_output_cols + channel*grad_output_rows*grad_output_cols 
              // + imageIndex*channels*grad_output_rows*grad_output_cols))<<std::endl;
              
            }
          }
          

        }
      }
      
      // for(int row = 0; row < input_rows -2*grad_index_y+1; row++)
      // {
      //   for(int col = 0; col < input_cols - 2*grad_index_x +1; col++)
      //   {
      //     *(output_arr + col + row*colsFinal + channel*rowsFinal*colsFinal + imageIndex*channels*rowsFinal*colsFinal) 
      //     = *(output_padded_arr + col + 2*grad_index_x + (row+2*grad_index_y)*colsPadded + channel*rowsPadded*colsPadded + imageIndex*channels*rowsPadded*colsPadded);
      //   }
        
      // }
      // end of channel
      // for(int row = 0; row < rowsFinal; row++)
      // {
      //   for(int col = 0; col < colsFinal; col++)
      //   {
      //     std::cout << *(output_arr + col + row*colsFinal + channel*rowsFinal*colsFinal + imageIndex*channels*rowsFinal*colsFinal) << " ";
          
      //   }
      //   std::cout << std::endl;
        
      // }

    }
  }

  
  torch::Tensor new_output = torch::from_blob(output_arr, {batch, channels, rowsFinal, colsFinal});

  return new_output.clone(); 
  


  
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &splatter_forward, "splatter forward");
  m.def("forward_non_sparse", &splatter_forward_non_sparse, "splatter forward non sparse");
  m.def("backward", &splatter_backward, "splatter backward");
  m.def("backward_input", &splatter_backward_input, "splatter backward input");
  m.def("backward_filter", &splatter_backward_filter, "splatter backward filter");
}