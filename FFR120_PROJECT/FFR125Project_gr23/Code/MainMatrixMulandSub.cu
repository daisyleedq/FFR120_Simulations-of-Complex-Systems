//
// Created by daisy on 15.11.17.
//
#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "device_launch_parameters.h"
#include "MatrixRW.h"
#include "MatrixMulHost.h"

#define  TILE_WIDTH 4

const int CrowN=16;//16 agents
const int CcolN=4*16;//4 neurons,16 is populationSize. could calculate whole population's output together

const int ArowN=CrowN;
const int AcolN=16*3+1*3;//16 preys,1 predator

const int BrowN=AcolN;
const int BcolN=CcolN;

const int gridY=(CrowN+TILE_WIDTH-1)/TILE_WIDTH;
const int gridX=(CcolN+TILE_WIDTH-1)/TILE_WIDTH;


int main(void){
    float* A;
    float* B;
    float* C;

    A= ( float * ) malloc ( ArowN * AcolN * sizeof ( float ));
    B = ( float *) malloc ( BrowN * BcolN * sizeof ( float ));
    C = ( float *) malloc ( CrowN * CcolN * sizeof ( float ));

    ReadMatrix (A, width );
    ReadMatrix (N, width );
    MatrixMulHost(A, B, C, ArowN, AcolN,  BrowN,  BcolN,  CrowN, CcolN , TILE_WIDTH );
    printf ("\n");
    PrintMatrix (M, width );
    printf ("\n");
    PrintMatrix (N, width );
    printf ("\n");
    PrintMatrix (C, width );
    free (M);
    free (N);
    free (C);
}