#include <stdlib.h>
#include <stdio.h>


//correlation forward
void splatterForward(int rows, int cols, int kernelSize,double **input, double **output, double **kernel ){
/*
currently only works for odd kernels
*/
    
    int weightIndex = kernelSize -2;

   

    //rows
    for(int i = weightIndex; i < weightIndex+rows; i++){
        //cols
        for(int j = weightIndex; j < weightIndex+cols; j++){
            
            //check if nonzero
            // float temp = input[i][j];
            if(input[i][j] != 0){
                
                for(int k = -1*weightIndex; k <= weightIndex; k++){
                    for(int l = -1*weightIndex; l <= weightIndex; l++){
                        
                        // printf("%d %d %d %d\n", i, j, k, l);
                        // float kern = kernel[k][l];
                        output[i-k][j-l] += kernel[1+k][1+l] * input[i][j];
                        
                        
                        // printf("%d %d %f\n", i, j, input[i][j]);
                        // printf("%d %d %f\n", k, l, kernel[1+k][1+l]);
                        // printf("%f\n\n", output[i][j]);

                        
                    }
                }
            }
        }
    }

}




//correlation forward
void splatterBackwardInput(int rows, int cols, int kernelSize,double **input, double **output, double **kernel ){
/*
currently only works for odd kernels
*/
    
    int weightIndex = kernelSize/2;

   

    //rows
    for(int i = weightIndex; i < weightIndex+rows; i++){
        //cols
        for(int j = weightIndex; j < weightIndex+cols; j++){
            
            //check if nonzero
            // float temp = input[i][j];
            if(input[i][j] != 0){
                
                for(int k = -1*weightIndex; k <= weightIndex; k++){
                    for(int l = -1*weightIndex; l <= weightIndex; l++){
                        
                        // printf("%d %d %d %d\n", i, j, k, l);
                        // float kern = kernel[k][l];
                        output[i+k][j+l] += kernel[weightIndex+k][weightIndex+l] * input[i][j];
                        
                        
                        // printf("%d %d %f\n", i, j, input[i][j]);
                        // printf("%d %d %f\n", k, l, kernel[weightIndex+k][weightIndex+l]);
                        // printf("%f\n\n", output[i][j]);

                        
                    }
                }
            }
        }
    }

}



//correlation forward
void splatterBackwardFilter(int rows, int cols, int kernelSize,double **input, double **output, double **kernel ){
/*
currently only works for odd kernels
*/
    
    int weightIndex = kernelSize/2;

   

    //rows
    for(int i = weightIndex; i < weightIndex+rows; i++){
        //cols
        for(int j = weightIndex; j < weightIndex+cols; j++){
            
            //check if nonzero
            // float temp = input[i][j];
            
            if(input[i][j] != 0){
                
                for(int k = -1*weightIndex; k < weightIndex; k++){
                    for(int l = -1*weightIndex; l < weightIndex; l++){
                        
                        // printf("%d %d %d %d\n", i, j, k, l);
                        // float kern = kernel[k][l];
                        output[i-k][j-l] += kernel[weightIndex+k][weightIndex+l] * input[i][j];
                        
                        
                        // printf("%d %d %f\n", i, j, input[i][j]);
                        // printf("%d %d %f\n", k, l, kernel[weightIndex+k][weightIndex+l]);
                        // printf("%f\n\n", output[i][j]);

                        
                    }
                }
            }
        }
    }

}