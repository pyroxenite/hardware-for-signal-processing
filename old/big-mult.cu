// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>
// #include "matrix.h"

// int main()
// {
//     srand(time(NULL));

//     clock_t start, end;
//     double time_used;
//     double gpu_tot = 0;

//     uint m = 10, n = 10, p = 10;

//     do
//     {
//         printf("\n############# Testing m = %d, n = %d, p = %d #############\n", m, n, p);
//         start = clock();

//         float* mat1 = randomIntegerMatrix(m, n, 5);
//         float* mat2 = randomIntegerMatrix(n, p, 5);
//         float* result = randomIntegerMatrix(m, p, 1);
//         end = clock();
//         time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//         printf("Matrix init : %f s \n", time_used);

//         start = clock();

//         float* mat1_gpu;
//         float* mat2_gpu;
//         float* result_gpu;
//         cudaMalloc(&mat1_gpu, sizeof(float) * m * n);
//         cudaMalloc(&mat2_gpu, sizeof(float) * n * p);
//         cudaMalloc(&result_gpu, sizeof(float) * m * p);
//         end = clock();
//         time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//         gpu_tot += time_used;
//         printf("cudamalloc : %f s \n", time_used);

//         start = clock();
//         cudaMemcpy(mat1_gpu, mat1, sizeof(float) * m * n, cudaMemcpyHostToDevice);
//         cudaMemcpy(mat2_gpu, mat2, sizeof(float) * n * p, cudaMemcpyHostToDevice);
//         end = clock();
//         time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//         gpu_tot += time_used;
//         printf("cudamemecopy : %f s \n", time_used);

//         start = clock();
//         matrixMult<<<m, p>>>(mat1_gpu, mat2_gpu, result_gpu, m, n, p);
//         end = clock();
//         time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//         gpu_tot += time_used;
//         printf("gpu mult : %f s \n", time_used);
//         printf("gpu tot : %f s \n", gpu_tot);

//         cudaMemcpy(result, result_gpu, sizeof(float) * m * p, cudaMemcpyDeviceToHost);

//         start = clock();
//         matrixMultCPU(mat1, mat2, result, m, n, p);
//         end = clock();
//         time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//         printf("cpu mult : %f s \n", time_used);
//         cudaDeviceSynchronize();

//         m *= 1.1;
//         n *= 1.1;
//         p *= 1.1;

//     } while (time_used < gpu_tot);

//     return 0;
// }