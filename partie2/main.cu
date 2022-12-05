#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix-new.h"
#include <math.h>

// int main() {
//     srand(time(NULL));

//     FloatMatrix* raw_data = randomMatrix(32, 32);

//     FloatMatrix** kernals = randomMatrices(6, 5, 5);
//     FloatMatrix** postConvolution = zeroMatrices(6, 28, 28);
//     FloatMatrix** postSubsampling = zeroMatrices(6, 14, 14);

//     convolve(raw_data, kernals[0], postConvolution[0]);

//     freeMatrix(raw_data);
//     forEach(kernals, 6, freeMatrix);
//     forEach(postConvolution, 6, freeMatrix);
//     forEach(postSubsampling, 6, freeMatrix);

//     cudaDeviceSynchronize();
//     return 0;
// }

int main() {
    srand(time(NULL));

    FloatMatrix* image = zeroMatrix(16, 16);
    FloatMatrix* kernal = zeroMatrix(5, 5);
    FloatMatrix* result = zeroMatrix(12, 12);

    drawCircle(image, 7.5, 7.5, 6, 1);
    drawCircle(kernal, 2, 2, 2.5, 1);

    displayMatrixAsAscii(image);
    displayMatrixAsAscii(kernal);

    convolve(image, kernal, result);

    cudaDeviceSynchronize();

    printMatrix(result);

    return 0;
}