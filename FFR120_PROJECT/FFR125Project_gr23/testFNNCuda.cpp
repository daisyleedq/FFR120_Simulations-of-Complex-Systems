//
// Created by daisy on 14.11.17.
//

//population=1000
//inputs:3,outputs:1000*2,weights1:1000*3*5*,weights2=5*2*1000,T1=1*5*1000,T2=1*2*1000,hiddenNeuron=1000*5


#pragma once

#include <host_defines.h>
#include <cuda.h>
#include <device_launch_parameters.h>

//it seems has to use shared memory
const int THREAD_NUM=16;
const int BLOCK_NUM=64;

__global__ static void kernelFNN(float* outputs, float* hiddenNeurons, const float* input, const float* weightsOne, const float* weightsTwo, const float* T1, const float* T2 ){
    const int tid=threadIdx.x;
    const int bid=blockIdx.x;

    const int idx=bid*THREAD_NUM+tid;//idx individual
    
    int i=0;
    for (i;i<5;i++){
        hiddenNeurons[i]=
    }
}

