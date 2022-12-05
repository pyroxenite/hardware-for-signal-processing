#include "matrix.h"

__host__ float* randomIntegerMatrix(int n, int m, int max) {
    float* mat = (float*) malloc(sizeof(int) * n * m);
    for (int i=0; i<n*m; i++) {
        mat[i] = rand() % max;
    }
    return mat;
}

__host__ float* randomMatrix(int n, int m) {
    int max = 100000;
    float* mat = (float*) malloc(sizeof(int) * n * m);
    for (int i=0; i<n*m; i++) {
        mat[i] = (rand() % max) / (float) max;
    }
    return mat;
}

__host__ void printMatrix(float* mat, int n, int m) {
    int l = n*m;
    printf("Matrix([\n");
    for (int i=0; i<l; i++) {
        if (i % m == 0) {
            printf("  [ %4.1f,", mat[i]);
        } else if (i % m == m-1) {
            printf(" %4.1f ],\n", mat[i]);
        } else {
            printf(" %4.1f,", mat[i]);
        }
    }
    printf("])\n");
}

__host__ void displayMatrixAsAscii(float* mat, int n, int m) {
    char levels[] = " .:;+=xX$&";
    int l = n*m;
    printf("@@@@");
    for (int i=0; i<m+2; i++)
        printf("@@");
    printf("\n@@");
    for (int i=0; i<m+2; i++)
        printf("  ");
    printf("@@\n@@  ");
    for (int i=0; i<l; i++) {
        float val = mat[i];
        int lev = (int) (val * 10);
        if (lev > 9) lev = 9;
        if (lev < 0) lev = 0;
        printf("%c%c", levels[lev], levels[lev]);
        if (i % m == m-1) {
            printf("  @@\n@@  ");
        }
    }

    for (int i=0; i<m+1; i++)
        printf("  ");
    printf("@@\n@@");
    for (int i=0; i<m+2; i++)
        printf("@@");
    printf("@@\n");
}

__host__ void matrixMultCPU(float* mat1, float* mat2, float* result, int m, int n, int p) {
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

__global__ void matrixMult(float* mat1, float* mat2, float* result, int m, int n, int p) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    result[i*blockDim.x + j] = 0;
    for (int k=0; k<n; k++) {
        result[i*blockDim.x + j] += mat1[i*m + k] * mat2[k*n + j];
    }
}

__global__ void matrixMult2(float* mat1, float* mat2, float* result, int m, int n, int p) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    result[i*blockDim.x + j] = 0;
    for (int k=0; k<n; k++) {
        result[i*blockDim.x + j] += mat1[i*m + k] * mat2[k*n + j];
    }
}

