#include "matrix-new.h"

__host__ FloatMatrix* newMatrix(float* cpu, int m, int n) {
    FloatMatrix* matrix = (FloatMatrix*) malloc(sizeof(FloatMatrix));
    matrix->cpu = cpu;
    matrix->m = m;
    matrix->n = n;
    cudaMalloc(&(matrix->gpu), sizeof(float) * m * n);
    copyToDevice(matrix);
    return matrix;
}

__host__ void freeMatrix(FloatMatrix* matrix) {
    free(matrix->cpu);
    cudaFree(matrix->gpu);
    free(matrix);
}

__host__ FloatMatrix* zeroMatrix(int m, int n) {
    FloatMatrix* matrix = newMatrix(
        (float*) calloc(sizeof(float), m * n),
        m,
        n
    );
    copyToDevice(matrix);
    return matrix;
}

__host__ FloatMatrix** zeroMatrices(int count, int m, int n) {
    FloatMatrix** matrices = (FloatMatrix**) malloc(count * sizeof(FloatMatrix*));
    for (int i=0; i<count; i++) {
        matrices[i] = zeroMatrix(m, n);
    }
    return matrices;
}

__host__ FloatMatrix* randomMatrix(int n, int m) {
    float* cpuMatrix = (float*) malloc(sizeof(float) * m * n);
    for (int i=0; i<m*n; i++) {
        cpuMatrix[i] = (rand() % RAND_MAX) / (float) RAND_MAX;
    }
    return newMatrix(cpuMatrix, n, m);
}

__host__ FloatMatrix** randomMatrices(int count, int m, int n) {
    FloatMatrix** matrices = (FloatMatrix**) malloc(count * sizeof(FloatMatrix*));
    for (int i=0; i<count; i++) {
        matrices[i] = randomMatrix(m, n);
    }
    return matrices;
}

__host__ void forEach(FloatMatrix** matrices, int count, void (*fun)(FloatMatrix* matrix)) {
    for (int i=0; i<count; i++) {
        fun(matrices[i]);
    }
}

__host__ void copyToDevice(FloatMatrix* matrix) {
    cudaMemcpy(
        matrix->gpu, 
        matrix->cpu, 
        sizeof(float)*matrix->m*matrix->n, 
        cudaMemcpyHostToDevice
    );
}

__host__ void copyFromDevice(FloatMatrix* matrix) {
    cudaMemcpy(
        matrix->cpu, 
        matrix->gpu, 
        sizeof(float)*matrix->m*matrix->n, 
        cudaMemcpyDeviceToHost
    );
}

__host__ void printMatrix(FloatMatrix* matrix) {
    int l = matrix->n * matrix->m;
    printf("Matrix([\n");
    for (int i=0; i<l; i++) {
        if (i % matrix->m == 0) {
            printf("  [ %4.1f,", matrix->cpu[i]);
        } else if (i % matrix->m == matrix->m-1) {
            printf(" %4.1f ],\n", matrix->cpu[i]);
        } else {
            printf(" %4.1f,", matrix->cpu[i]);
        }
    }
    printf("])\n");
}

__host__ void displayMatrix(FloatMatrix* matrix) {
    char levels[] = " .:;+=xX$&";
    int l = matrix->n * matrix->m;
    printf("@@@@");
    for (int i=0; i<matrix->m+2; i++)
        printf("@@");
    printf("\n@@");
    for (int i=0; i<matrix->m+2; i++)
        printf("  ");
    printf("@@\n@@  ");
    for (int i=0; i<l; i++) {
        float val = matrix->cpu[i];
        int lev = (int) (val * 10);
        if (lev > 9) lev = 9;
        if (lev < 0) lev = 0;
        printf("%c%c", levels[lev], levels[lev]);
        if (i % matrix->m == matrix->m-1) {
            printf("  @@\n@@  ");
        }
    }

    for (int i=0; i<matrix->m+1; i++)
        printf("  ");
    printf("@@\n@@");
    for (int i=0; i<matrix->m+2; i++)
        printf("@@");
    printf("@@\n");
}

__host__ void displaySignedMatrix(FloatMatrix* matrix) {
    char levels[] = " .:;+=xX$&";
    int l = matrix->n * matrix->m;
    printf("%s@@@@", RESET);
    for (int i=0; i<matrix->m+2; i++)
        printf("@@");
    printf("\n@@");
    for (int i=0; i<matrix->m+2; i++)
        printf("  ");
    printf("%s@@\n@@  ", RESET);
    for (int i=0; i<l; i++) {
        float val = matrix->cpu[i];

        int lev = (int) (abs(val) * 10);
        if (lev > 9) lev = 9;
        if (lev < 0) lev = 0;
        if (val > 0)
            printf("%s%c%c", KBLU, levels[lev], levels[lev]);
        else 
            printf("%s%c%c", KRED, levels[lev], levels[lev]);
        if (i % matrix->m == matrix->m-1) {
            printf("%s  @@\n@@  ", RESET);
        }
    }

    for (int i=0; i<matrix->m+1; i++)
        printf("  ");
    printf("%s@@\n@@", RESET);
    for (int i=0; i<matrix->m+2; i++)
        printf("@@");
    printf("@@\n");
}

__global__ void convolveGpu(float* image, float* kernal, float* result, int im_m, int im_n, int ker_m, int ker_n) {
    int res_i = threadIdx.x;
    int res_j = blockIdx.x;
    int ker_i, ker_j;
    int im_i, im_j;
    float sum = 0;
    for (int i=0; i<ker_m*ker_n; i++) {
        ker_i = i / ker_n;
        ker_j = i % ker_n;
        im_i = res_i + ker_i;
        im_j = res_j + ker_j;
        sum += image[im_i*im_n + im_j] * kernal[ker_i*ker_n + ker_j];
    }
    result[res_i*blockDim.x + res_j] = sum;
}

__host__ void convolve(FloatMatrix* image, FloatMatrix* kernal, FloatMatrix* result) {
    convolveGpu<<<image->m - kernal->m + 1, image->n - kernal->n + 1>>>(
        image->gpu, kernal->gpu, result->gpu, image->m, image->n, kernal->m, kernal->n
    );
}

__global__ void drawCircleGpu(float* image, float x, float y, float r, float color) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    int n = blockDim.x;
    if (sqrt((i-y)*(i-y) + (j-x)*(j-x)) < r) {
        image[i*n + j] = color;
    }
}

__host__ void drawCircle(FloatMatrix* matrix, float x, float y, float r, float color) {
    drawCircleGpu<<<matrix->n, matrix->m>>>(matrix->gpu, x, y, r, color);
}

__global__ void subsampleGpu(float* input, float* output, int amount) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    int n = blockDim.x;
    output[i*n + j] = input[i*n*amount*amount + j*amount];
}

__host__ void subsample(FloatMatrix* input, FloatMatrix* output, int amount) {
    subsampleGpu<<<output->n, output->m>>>(input->gpu, output->gpu, amount);
}

__global__ void applyActivationGpu(float* matrix) {
    int i = threadIdx.x;
    int j = blockIdx.x;
    int n = blockDim.x;
    matrix[i*n + j] = tanh(matrix[i*n + j]);
}

__host__ void applyActivation(FloatMatrix* matrix) {
    applyActivationGpu<<<matrix->n, matrix->m>>>(matrix->gpu);
}

__global__ void matrixMultGpu(float* mat1, float* mat2, float* result, int m, int n, int p) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    result[i*blockDim.x + j] = 0;
    for (int k=0; k<n; k++) {
        result[i*blockDim.x + j] += mat1[i*m + k] * mat2[k*n + j];
    }
}

__host__ void matrixMult(FloatMatrix* mat1, FloatMatrix* mat2, FloatMatrix* result) {
    matrixMultGpu<<<result->n, result->m>>>(mat1->gpu, mat2->gpu, result->gpu, mat1->m, mat2->m, mat2->n);
}

__host__ FloatMatrix* loadMatrix(const char* filename, int m, int n) {
    float* matrix = (float*) malloc(sizeof(float) * m * n);
    FILE* file = fopen(filename, "rb");
    fread((void*) matrix, sizeof(float), m * n, file);
    fclose(file);
    return newMatrix(matrix, m, n);
}

__host__ FloatMatrix** loadMatrices(const char* filename, int count, int m, int n) {
    FloatMatrix** matrices = zeroMatrices(count, m, n);
    FILE* file = fopen(filename, "rb");
    for (int i=0; i<count; i++) {
        fread((void*) matrices[i]->cpu, sizeof(float), m * n, file);
    }
    forEach(matrices, count, copyToDevice);
    fclose(file);
    return matrices;
}