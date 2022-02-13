from random import weibullvariate
import numpy as np

def splat_conv2d(input, kernel, mode=''):
    #this function skips reversing the kernel
    #establish size of input image
    # print("starting convolution")
    rows = input.shape[0]
    cols = input.shape[1]

    #establish weight kernel size
    weightSize = kernel.shape[0]

    #establish weight indexing
    odd = False
    weightIndex = int(weightSize/2) 
    if(weightSize%2):
        odd = True
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
            return output
        else:
            return output[:-1, :-1]
    elif(mode == "valid"):
        if(cols<weightSize or rows< weightSize):
            if(odd): 
                return output[rows-1:-rows+1, cols-1:-cols+1]
            else:
                return output[rows-1:-rows, cols-1:-cols]
        if(odd):
            return output[2*weightIndex:-2*weightIndex, 2*weightIndex:-2*weightIndex]
        else:
            return output[2*weightIndex-1:-2*weightIndex ,2*weightIndex-1:-2*weightIndex]
    output = output[weightIndex:-weightIndex, weightIndex:-weightIndex]
    
    return output
    #end of function

def splat_corr2d(input, kernel, mode=''):
   

   
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
        return output
    elif(mode == "valid"):
        if(odd):
            return output[2*weightIndex:-2*weightIndex, 2*weightIndex:-2*weightIndex]
        else:
            return output[2*weightIndex-1:-2*weightIndex ,2*weightIndex-1:-2*weightIndex]
    output = output[weightIndex:-weightIndex, weightIndex:-weightIndex]
    
    return output
    #end of function