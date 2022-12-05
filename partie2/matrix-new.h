#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>

typedef struct floatMatrix {
    float* cpu; // optional
    float* gpu;
    int m;
    int n;
} FloatMatrix;

__host__ FloatMatrix* newMatrix(float* cpu, int n, int m);

__host__ void freeMatrix(FloatMatrix* matrix);

__host__ FloatMatrix* zeroMatrix(int m, int n);

__host__ FloatMatrix** zeroMatrices(int count, int m, int n);

__host__ FloatMatrix* randomMatrix(int n, int m);

__host__ FloatMatrix** randomMatrices(int count, int n, int m);

__host__ void forEach(FloatMatrix** matrices, int count, void (*fun)(FloatMatrix* matrix));

__host__ void copyToDevice(FloatMatrix* matrix);

__host__ void copyFromDevice(FloatMatrix* matrix);

__host__ void printMatrix(FloatMatrix* matrix);

__host__ void displayMatrixAsAscii(FloatMatrix* matrix);

__host__ void convolve(FloatMatrix* image, FloatMatrix* kernal, FloatMatrix* result);

__host__ void drawCircle(FloatMatrix* matrix, float x, float y, float r, float color);