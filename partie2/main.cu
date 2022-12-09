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

void blurDemo() {
    int im_size = 31;
    int ker_size = 6;
    int res_size = im_size - ker_size + 1;

    // Initialize some matrices.
    FloatMatrix* image = zeroMatrix(im_size, im_size);
    FloatMatrix* kernal = zeroMatrix(ker_size, ker_size);
    FloatMatrix* result = zeroMatrix(res_size, res_size);
    FloatMatrix* subsampledResult = zeroMatrix(res_size/2, res_size/2);

    // Create example input image. (Operation done on GPU side.)
    drawCircle(image, im_size/2.5, im_size/2.5, im_size/5.0, 0.4);
    drawCircle(image, 1.5*im_size/2.5, 1.5*im_size/2.5, im_size/5.0, 0.1);

    // Create example kernal. (Operation done on GPU side.)
    drawCircle(kernal, ker_size/2 - 0.5, ker_size/2 - 0.5, ker_size/2, 0.1);

    // Apply convolution.
    convolve(image, kernal, result);

    // Subsample by a factor of 2.
    subsample(result, subsampledResult, 2);

    // Wait for GPU.
    cudaDeviceSynchronize();

    // Copy data from GPU.
    copyFromDevice(image);
    copyFromDevice(kernal);
    copyFromDevice(result);
    copyFromDevice(subsampledResult);

    // Display matrices.
    displayMatrix(image);
    displayMatrix(kernal);
    displayMatrix(result);
    displayMatrix(subsampledResult);

    // Free allocated memory.
    freeMatrix(image);
    freeMatrix(kernal);
    freeMatrix(result);
}


void sobelDemo() {
    int im_size = 32;
    int ker_size = 3;
    int res_size = im_size - ker_size + 1;

    // Initialize some matrices.
    FloatMatrix* image = zeroMatrix(im_size, im_size);
    FloatMatrix* kernal = zeroMatrix(ker_size, ker_size);
    FloatMatrix* result = zeroMatrix(res_size, res_size);

    // Draw an example input image. This is done on the GPU side.
    drawCircle(image, im_size/3, im_size/3, im_size/3.8, 0.4);
    drawCircle(image, 2*im_size/3, 2*im_size/3, im_size/3.8, 0.1);

    // Create a kernal on the CPU then copy to GPU.
    for (int i=0; i<ker_size; i++) {
        for (int j=0; j<ker_size; j++) {
            kernal->cpu[i*ker_size + j] = (j - ker_size/2.0 + 0.5)/ker_size*2/(1 + abs(i - ker_size/2.0 + 0.5));
        }
    }
    copyToDevice(kernal);
    
    // Apply convolution.
    convolve(image, kernal, result);

    // Wait for GPU.
    cudaDeviceSynchronize();
    
    // Copy data from GPU.
    copyFromDevice(image);
    copyFromDevice(kernal);
    copyFromDevice(result);

    // Display matrices.
    displayMatrix(image);
    displaySignedMatrix(kernal); // negative -> red, positive -> blue
    displaySignedMatrix(result);

    // Free allocated memory.
    freeMatrix(image);
    freeMatrix(kernal);
    freeMatrix(result);
}


int main() {
    srand(time(NULL));

    sobelDemo();

    //blurDemo();
    
    return 0;
}