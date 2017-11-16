//
// Created by daisy on 15.11.17.
//

#include "cuda.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "device_launch_parameters.h"

#define  TILE_WIDTH 2

//simple example
//1. N*N matrix for ALL M, N, P
//2. No shared memory, low efficiency

__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int width){
    //blockDim gives number of threads in each block in this direction
    float  cValue=0.0;

    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;

    if (row >width || col >width) return;

    for (int i=0;i<width;i++){
        cValue+=(Md[row*width+i]*Nd[i*width+col]);
    }
    Pd[row*width+col]=cValue;
}

void MatrixMultiplication(float* M, float* N, float* P,int width){
    int size=width*width*sizeof(float);
    float * Md;
    float * Nd;
    float * Pd;

    cudaMalloc((void**) &Md,size);
    cudaMalloc((void**) &Nd,size);
    cudaMalloc((void**) &Pd,size);

    cudaMemcpy(Md,M,size,cudaMemcpyHostToDevice);
    cudaMemcpy(Nd,N,size,cudaMemcpyHostToDevice);

    dim3 dimGrid((width+TILE_WIDTH-1)/TILE_WIDTH,(width+TILE_WIDTH-1)/TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);

    MatrixMulKernel<<<dimGrid,dimBlock>>>(Md,Nd,Pd,width);

    cudaMemcpy(P,Pd,size,cudaMemcpyDeviceToHost);

    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
}

void ReadMatrix(float* M,int width){
    int i=0;
    int j=0;
    for(i=0;i<width;i++){
        for(j=0;j<width;j++){
            M[i*width+j]=rand();
        }
    }
}

void PrintMatrix(float*M,int width){
    int i;
    int j;
    for (i = 0; i < width ; i ++) {
        for (j = 0; j < width ; j ++) {
            printf ("%f␣", M[i * width + j]);
            }
            printf ("\n");
        }
}

int main(void){
    float* M;
    float* N;
    float* P;
    int width=32;
//    scanf("%d",&width);

    M = ( float * ) malloc ( width * width * sizeof ( float ));
    N = ( float *) malloc ( width * width * sizeof ( float ));
    P = ( float *) malloc ( width * width * sizeof ( float ));

    ReadMatrix (M, width );
    ReadMatrix (N, width );
    MatrixMultiplication (M, N, P, width );
    PrintMatrix (M, width );
    printf ("\n");
    PrintMatrix (N, width );
    printf ("\n");
    PrintMatrix (P, width );
    free (M);
    free (N);
    free (P);
}