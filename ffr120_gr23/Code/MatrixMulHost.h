//
// Created by daisy on 15.11.17.
//

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "cuda.h"
#include "cuda_runtime.h"
#include "stdio.h"

#endif //FFR125PROJECT_MATRIXMULHOST_H
__global__ void MatMul(float* A, float* B, float* C, const int ARows, const int ACols, const int BRows,
                       const int BCols, const int CRows, const int CCols, const int tileWidth);

__global__ void MatSub(float* A, float* B, float* C, const int rowN, int colN, const int tileWidth);

void MatrixMultHost(float* A, float* B, float* C,int ArowN, int AcolN, int BrowN, int BcolN, int CrowN, int CcolN ,int tileWidth);