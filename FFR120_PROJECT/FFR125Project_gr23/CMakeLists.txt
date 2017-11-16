cmake_minimum_required(VERSION 3.6)
project(FFR125Project)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp testFNNCuda.cpp MatrixMultiply.cu MatrixMulSharedRef.cpp MatrixRW.cpp MatrixRW.h MatrixMulHost.h MatrixMulHost.cu)
add_executable(FFR125Project ${SOURCE_FILES})