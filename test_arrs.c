#include <stdio.h>
#include <stdlib.h>


void test(int rows, int cols, double **arr){

    printf("rows: %d, cols: %d\n", rows, cols);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            printf("%f ", arr[i][j]);
        }
        printf("\n");
    }




}