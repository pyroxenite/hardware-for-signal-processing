#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

__host__ int* randomFlatMatrix(int n, int m, int max);

__host__ void printFlatMatrix(int* mat, int n, int m);

__host__ void matrixMultCPU(int* mat1, int* mat2, int* result, int m, int n, int p);

__global__ void matrixMult(int* mat1, int* mat2, int* result, int m, int n, int p);

__global__ void matrixMult2(int* mat1, int* mat2, int* result, int m, int n, int p);