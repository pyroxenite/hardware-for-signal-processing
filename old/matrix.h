#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

__host__ float* randomIntegerMatrix(int n, int m, int max);

__host__ float* zeroArray(int length);

__host__ float* randomArray(int length);

__host__ float* randomMatrix(int n, int m);

__host__ void printMatrix(float* mat, int n, int m);

__host__ void displayMatrixAsAscii(float* mat, int n, int m);

__host__ void matrixMultCPU(float* mat1, float* mat2, float* result, int m, int n, int p);

__global__ void matrixMult(float* mat1, float* mat2, float* result, int m, int n, int p);

__global__ void matrixMult2(float* mat1, float* mat2, float* result, int m, int n, int p);