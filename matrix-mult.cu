#include "matrix-mult.h"

__host__ int* randomFlatMatrix(int n, int m, int max) {
    int* mat = (int*) malloc(sizeof(int) * n * m);
    for (int i=0; i<n*m; i++) {
        mat[i] = rand() % max;
    }
    return mat;
}

__host__ void printFlatMatrix(int* mat, int n, int m) {
    int l = n*m;
    printf("Matrix([\n");
    for (int i=0; i<l; i++) {
        if (i % m == 0) {
            printf("  [ %2d,", mat[i]);
        } else if (i % m == m-1) {
            printf(" %2d ],\n", mat[i]);
        } else {
            printf(" %2d,", mat[i]);
        }
    }
    printf("])\n");
}

__host__ void matrixMultCPU(int* mat1, int* mat2, int* result, int m, int n, int p) {
    int l = n*m;
    int i, j;
    for (int c=0; c<l; c++) {
        i = c % m;
        j = c / m;
        result[i*m + j] = 0;
        for (int k=0; k<n; k++) {
            result[i*m + j] += mat1[i*m + k] * mat2[k*n + j];
        }
    }
}

__global__ void matrixMult(int* mat1, int* mat2, int* result, int m, int n, int p) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    result[i*blockDim.x + j] = 0;
    for (int k=0; k<n; k++) {
        result[i*blockDim.x + j] += mat1[i*m + k] * mat2[k*n + j];
    }
}

__global__ void matrixMult2(int* mat1, int* mat2, int* result, int m, int n, int p) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    result[i*blockDim.x + j] = 0;
    for (int k=0; k<n; k++) {
        result[i*blockDim.x + j] += mat1[i*m + k] * mat2[k*n + j];
    }
}

