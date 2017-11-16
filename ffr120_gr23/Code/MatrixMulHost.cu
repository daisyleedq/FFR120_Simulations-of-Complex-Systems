//
// Created by daisy on 15.11.17.
//

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "cuda.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "device_launch_parameters.h"

__global__ void MatMul(float* A, float* B, float* C, const int ARows, const int ACols, const int BRows,
                       const int BCols, const int CRows, const int CCols, const int tileWidth)
{
    float CValue = 0;

    int Row = blockIdx.y*tileWidth + threadIdx.y;
    int Col = blockIdx.x*tileWidth + threadIdx.x;

    __shared__ float As[tileWidth][tileWidth];
    __shared__ float Bs[tileWidth][tileWidth];

    for (int k = 0; k < (tileWidth + ACols - 1)/tileWidth; k++) {

        if (k*tileWidth + threadIdx.x < ACols && Row < ARows)
            As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*tileWidth + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x]=0.0;

        if (k*tileWidth + threadIdx.y < BRows && Col < BCols)
            Bs[threadIdx.y][threadIdx.x] = B[(k*tileWidth + threadIdx.y)*BCols + Col];
        else
            Bs[threadIdx.y][threadIdx.x]=0.0;


        __syncthreads();

        for (int n = 0; n < tileWidth; ++n)
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

        __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
          (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}

__global__ void MatSub(float* A, float* B, float* C, const int rowN, int colN, const int tileWidth)
{

    int i = blockIdx.y*tileWidth + threadIdx.y;//which row
    int j = blockIdx.x*tileWidth + threadIdx.x;//which column
    C[i*colN+j]=A[i*colN+j]-B[i*colN+j];

}


//TOD: realise here or seperate and call for FNN


//allocate memory
void MatrixMultHost(float* A, float* B, float* C,int ArowN, int AcolN, int BrowN, int BcolN, int CrowN, int CcolN ,int tileWidth){

    if(AcolN!=BrowN || CrowN!=ArowN || CcolN!=BcolN){
        printf("The intput matrix don't match for multiplication\n");
    }
    int sizeAd=ArowN*AcolN*sizeof(float);
    int sizeBd=AcolN*BcolN*sizeof(float);
    int sizeCd=CrowN*CcolN*sizeof(float);
    //memory on device
    float * Ad;
    float * Bd;
    float * Cd;

    cudaMalloc((void**)&Ad,sizeAd);
    cudaMalloc((void**)&Bd,sizeBd);
    cudaMalloc((void**)&Cd,sizeCd);

    cudaMemcpy(Ad,A,sizeAd,cudaMemcpyHostToDevice);
    cudaMemcpy(Bd,B,sizeBd,cudaMemcpyHostToDevice);

    //dim3 dimBlock(width,width,1);
    //dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);

    int gridY=(CrowN+tileWidth-1)/tileWidth;
    int gridX=(CcolN+tileWidth-1)/tileWidth;

    dim3 dimGrid(gridY,gridX);
    dim3 dimBlock(tileWidth,tileWidth);

    MatMul<<<dimGrid,dimBlock>>>(Ad,Bd,Cd, ArowN, AcolN, BrowN,
            BcolN, CrowN, CcolN,tileWidth);

    cudaMemcpy(C,Cd,sizeCd,cudaMemcpyDeviceToHost);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
}
