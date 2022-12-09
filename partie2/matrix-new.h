#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>

#define KRED  "\x1B[31m"
#define KBLU  "\x1B[34m"
#define RESET "\x1B[0m"

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

__host__ void displayMatrix(FloatMatrix* matrix);

__host__ void displaySignedMatrix(FloatMatrix* matrix);

__host__ void convolve(FloatMatrix* image, FloatMatrix* kernal, FloatMatrix* result);

__host__ void drawCircle(FloatMatrix* matrix, float x, float y, float r, float color);

__host__ void subsample(FloatMatrix* input, FloatMatrix* output, int amount);

__host__ void applyActivation(FloatMatrix* matrix);