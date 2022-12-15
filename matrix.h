#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>

#ifndef __MATRIX__
#define __MATRIX__

#define KRED  "\x1B[31m"
#define KBLU  "\x1B[34m"
#define RESET "\x1B[0m"

#define COLUMN 1
#define ROW 0

typedef enum Activation { TANH, SOFTMAX } Activation;

typedef struct floatMatrix {
    float* cpu; // optional
    float* gpu;
    int m; // row count
    int n; // col count
} FloatMatrix;

__host__ FloatMatrix* newMatrix(float* cpu, int m, int n);

__host__ void freeMatrix(FloatMatrix* matrix);

__host__ FloatMatrix* zeroMatrix(int m, int n);

__host__ FloatMatrix** zeroMatrices(int count, int m, int n);

__host__ FloatMatrix* randomMatrix(int n, int m);

__host__ FloatMatrix** randomMatrices(int count, int n, int m);

__host__ void forEachMatrix(FloatMatrix** matrices, int count, void (*fun)(FloatMatrix* matrix));

__host__ void copyToDevice(FloatMatrix* matrix);

__host__ void copyFromDevice(FloatMatrix* matrix);

__host__ void printMatrix(FloatMatrix* matrix);

__host__ void displayMatrix(FloatMatrix* matrix);

__host__ void displaySignedMatrix(FloatMatrix* matrix);

__host__ void setMatrixToZero(FloatMatrix* matrix);

__host__ void addMatrix(FloatMatrix* matrix, FloatMatrix* sum);

__host__ void addValueToMatrix(FloatMatrix* matrix, float value);

__host__ void convolve(FloatMatrix* image, FloatMatrix* kernal, FloatMatrix* result);

__host__ void drawCircle(FloatMatrix* matrix, float x, float y, float r, float color);

__host__ void averagePool(FloatMatrix* input, FloatMatrix* output, int amount);

__host__ void applyActivation(FloatMatrix* matrix, Activation activation);

__host__ void flattenMatrices(FloatMatrix** matrices, int count, FloatMatrix* output);

__host__ void matrixMult(FloatMatrix* mat1, FloatMatrix* mat2, FloatMatrix* result);

__host__ FloatMatrix* loadMatrix(const char* filename, int m, int n);

__host__ FloatMatrix* loadVector(const char* filename, int n, int isColumn);

__host__ FloatMatrix** loadMatrices(const char* filename, int count, int m, int n);

#endif