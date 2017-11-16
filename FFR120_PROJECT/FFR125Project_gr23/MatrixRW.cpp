//
// Created by daisy on 15.11.17.
//
#include "cuda.h"
#include "stdio.h"
#include "MatrixRW.h"

void ReadMatrix(float* M,int row, int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            M[i*row+j]=rand();
        }
    }
}

void PrintMatrix(float* M,int row, int col){
    for (int i = 0; i < row ; i ++) {
        for (int j = 0; j < col ; j ++) {
            printf ("%.6f ", M[i * row + j]);
        }
        printf ("\n");
    }
}