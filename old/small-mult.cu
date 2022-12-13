// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>
// #include "matrix.h"

// int main() {
//     srand(time(NULL));

//     uint m = 4, n = 4, p = 4;

//     float* mat1 = randomIntegerMatrix(m, n, 5);
//     float* mat2 = randomIntegerMatrix(n, p, 5);
//     float* result = randomIntegerMatrix(m, p, 1);

//     float* mat1_gpu;
//     float* mat2_gpu;
//     float* result_gpu;
//     cudaMalloc(&mat1_gpu, sizeof(float)*m*n);
//     cudaMalloc(&mat2_gpu, sizeof(float)*n*p);
//     cudaMalloc(&result_gpu, sizeof(float)*m*p);

//     cudaMemcpy(mat1_gpu, mat1, sizeof(float)*m*n, cudaMemcpyHostToDevice);
//     cudaMemcpy(mat2_gpu, mat2, sizeof(float)*n*p, cudaMemcpyHostToDevice);

//     matrixMult<<<m,p>>>(mat1_gpu, mat2_gpu, result_gpu, m, n, p);

//     cudaMemcpy(result, result_gpu, sizeof(float)*m*p, cudaMemcpyDeviceToHost);

//     printf("mat1 = ");
//     printMatrix(mat1, m, n);
//     printf("mat2 = ");
//     printMatrix(mat2, n, p);
//     printf("result_gpu_1 = ");
//     printMatrix(result, m, p);

//     dim3 mp = { m, p, 1 };
//     matrixMult2<<<mp,1>>>(mat1_gpu, mat2_gpu, result_gpu, m, n, p);

//     cudaMemcpy(result, result_gpu, sizeof(float)*m*p, cudaMemcpyDeviceToHost);

//     printf("result_gpu_2 = ");
//     printMatrix(result, m, p);

//     matrixMultCPU(mat1, mat2, result, m, n, p);
//     printf("result_cpu = ");
//     printMatrix(result, m, p);

//     cudaDeviceSynchronize();
// 	return 0;
// }