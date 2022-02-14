from random import weibullvariate
import numpy as np

def splat_conv2d(large_input, kernel, mode=''):
    #this function skips reversing the kernel
    #establish size of input image
    # print("starting convolution")

    N = large_input.shape[0]
    channels = large_input.shape[1]
    height = large_input.shape[2]
    width = large_input.shape[3]

    sub = 0

    #establish weight kernel size
    weightSize = kernel.shape[0]

    #establish weight indexing
    odd = False
    weightIndex = int(weightSize/2) 
    if(weightSize%2):
        odd = True
    

    if(mode == "full"):
        if(odd):
            # output =  output
            pass
            sub -= 2
        else:
            # sub += 1
            sub -= 2
    elif(mode == "valid"):
        if(width<weightSize or height< weightSize):
            if(odd): 
                # output = output[rows-1:-rows+1, cols-1:-cols+1]
                sub = height
            else:
                # output =  output[rows-1:-rows, cols-1:-cols]
                sub = height
        if(odd):
            # output = output[2*weightIndex:-2*weightIndex, 2*weightIndex:-2*weightIndex]
            sub = 2*weightIndex
        else:
            # output =  output[2*weightIndex-1:-2*weightIndex ,2*weightIndex-1:-2*weightIndex]
            sub = 2*weightIndex + 1
    else: 
        # output = output[weightIndex:-weightIndex, weightIndex:-weightIndex]
        sub = weightIndex
    
            
    # print(N, channels, height, width, sub)
    large_output = np.zeros((N, channels, height-sub, width-sub ))
    # print(large_output.shape)

    for input_index, channel in enumerate(large_input):
        for channel_index, input in enumerate(channel):

            rows = input.shape[0]
            cols = input.shape[1]

            
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
                        if(odd):
                            output[i-weightIndex:i+weightIndex +1,j-weightIndex:j+weightIndex+1] += padInput[i][j]*kernel
                        else:
                            output[i-weightIndex:i+weightIndex,j-weightIndex:j+weightIndex] += padInput[i][j]*kernel


            #adjust final output size using splicing
            if(mode == "full"):
                if(odd):
                    output =  output
                else:
                    output =  output[:-1, :-1]
            elif(mode == "valid"):
                if(cols<weightSize or rows< weightSize):
                    if(odd): 
                        output = output[rows-1:-rows+1, cols-1:-cols+1]
                    else:
                        output =  output[rows-1:-rows, cols-1:-cols]
                if(odd):
                    output = output[2*weightIndex:-2*weightIndex, 2*weightIndex:-2*weightIndex]
                else:
                    output =  output[2*weightIndex-1:-2*weightIndex ,2*weightIndex-1:-2*weightIndex]
            else: 
                output = output[weightIndex:-weightIndex, weightIndex:-weightIndex]
            
            large_output[input_index][channel_index] = output
            #end of function

    return large_output

def splat_corr2d(large_input, large_kernel, mode=''):
   

    N = large_input.shape[0]
    channels = large_input.shape[1]
    height = large_input.shape[2]
    width = large_input.shape[3]

    sub = 0

    #establish weight kernel size
    weightSize = large_kernel.shape[0]
    if(len(large_kernel.shape) > 2):
        weightSize = large_kernel.shape[2]

    #establish weight indexing
    odd = False
    weightIndex = int(weightSize/2) 
    if(weightSize%2):
        odd = True
    

    if(mode == "full"):
        if(odd):
            # output =  output
            pass
        else:
            sub += 1
    elif(mode == "valid"):
        if(width<weightSize or height< weightSize):
            if(odd): 
                # output = output[rows-1:-rows+1, cols-1:-cols+1]
                sub = height
            else:
                # output =  output[rows-1:-rows, cols-1:-cols]
                sub = height
        if(odd):
            # output = output[2*weightIndex:-2*weightIndex, 2*weightIndex:-2*weightIndex]
            sub = 2*weightIndex
        else:
            # output =  output[2*weightIndex-1:-2*weightIndex ,2*weightIndex-1:-2*weightIndex]
            sub = 2*weightIndex -1
    else: 
        # output = output[weightIndex:-weightIndex, weightIndex:-weightIndex]
        sub = weightIndex
    
            
    # print(N, channels, height, width, sub, large_kernel.shape)
    large_output = np.zeros((N, channels, abs(height-sub), abs(width-sub) ))
    

    for input_index, channel in enumerate(large_input):
        for channel_index, input in enumerate(channel):

            kernel = large_kernel
            if(len(large_kernel.shape) > 2):
                kernel = large_kernel[input_index][channel_index]
            # else:
            #     kernel = large_kernel
   
            # print("starting convolution")
            kernel = np.flipud(np.fliplr(kernel))

            rows = input.shape[0]
            cols = input.shape[1]

            #establish weight kernel size
            weightSize = kernel.shape[0]

            odd = False
            if(weightSize%2):
                odd = True

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

                        #update output using vector operation
                        
                        if(odd):
                            output[i-weightIndex:i+weightIndex +1,j-weightIndex:j+weightIndex+1] += padInput[i][j]*kernel
                        else:
                            output[i-weightIndex:i+weightIndex ,j-weightIndex:j+weightIndex] += padInput[i][j]*kernel


           
            #adjust final output size using splicing
            if(mode == "full"):
                if(odd):
                    output =  output
                else:
                    output =  output[:-1, :-1]
            elif(mode == "valid"):
                if(cols<weightSize or rows< weightSize):
                    if(odd): 
                        output = output[rows-1:-rows+1, cols-1:-cols+1]
                    else:
                        output =  output[rows-1:-rows, cols-1:-cols]
                if(odd):
                    output = output[2*weightIndex:-2*weightIndex, 2*weightIndex:-2*weightIndex]
                else:
                    output =  output[2*weightIndex-1:-2*weightIndex ,2*weightIndex-1:-2*weightIndex]
            else: 
                output = output[weightIndex:-weightIndex, weightIndex:-weightIndex]
            
            large_output[input_index][channel_index] = output
            #end of function

    return large_output