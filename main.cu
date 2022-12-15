#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matrix.h"
#include "digit-classifier.h"

void blurDemo() {
    int im_size = 31;
    int ker_size = 6;
    int res_size = im_size - ker_size + 1;

    // Initialize some matrices. (CPU & GPU)
    FloatMatrix* image = zeroMatrix(im_size, im_size);
    FloatMatrix* kernal = zeroMatrix(ker_size, ker_size);
    FloatMatrix* result = zeroMatrix(res_size, res_size);
    FloatMatrix* subsampledResult = zeroMatrix(res_size/2, res_size/2);

    // Create example input image. (GPU)
    drawCircle(image, im_size/2.5, im_size/2.5, im_size/5.0, 0.4);
    drawCircle(image, 1.5*im_size/2.5, 1.5*im_size/2.5, im_size/5.0, 0.1);

    // Create example kernal. (GPU)
    drawCircle(kernal, ker_size/2 - 0.5, ker_size/2 - 0.5, ker_size/2, 0.1);

    // Apply convolution. (GPU)
    convolve(image, kernal, result);

    // Subsample by a factor of 2. (GPU)
    averagePool(result, subsampledResult, 2);

    // Wait for GPU.
    cudaDeviceSynchronize();

    // Copy data from GPU to CPU.
    copyFromDevice(image);
    copyFromDevice(kernal);
    copyFromDevice(result);
    copyFromDevice(subsampledResult);

    // Display matrices. (CPU)
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

    // Draw an example input image. (GPU)
    drawCircle(image, im_size/3, im_size/3, im_size/3.8, 0.4);
    drawCircle(image, 2*im_size/3, 2*im_size/3, im_size/3.8, 0.1);

    // Create a kernal on the CPU then copy to GPU.
    for (int i=0; i<ker_size; i++) {
        for (int j=0; j<ker_size; j++) {
            kernal->cpu[i*ker_size + j] = (j - ker_size/2.0 + 0.5)/ker_size*2/(1 + abs(i - ker_size/2.0 + 0.5));
        }
    }
    copyToDevice(kernal);
    
    // Apply convolution. (GPU)
    convolve(image, kernal, result);

    // Wait for GPU.
    cudaDeviceSynchronize();
    
    // Copy data from GPU to CPU.
    copyFromDevice(image);
    copyFromDevice(kernal);
    copyFromDevice(result);

    // Display matrices. (CPU)
    displayMatrix(image);
    displaySignedMatrix(kernal); // negative -> red, positive -> blue
    displaySignedMatrix(result);

    // Free allocated memory.
    freeMatrix(image);
    freeMatrix(kernal);
    freeMatrix(result);
}

void paramsReadTest() {
    // Read matrices from files.
    FloatMatrix** kernals = loadMatrices("data/conv1-weights.bin", 6, 5, 5);
    FloatMatrix* bias = loadVector("data/conv1-bias.bin", 6, COLUMN);

    // Display them.
    forEachMatrix(kernals, 6, displaySignedMatrix);
    displaySignedMatrix(bias);
    
    // Free allocated memory.
    forEachMatrix(kernals, 6, freeMatrix);
    freeMatrix(bias);
}

void imageReadTest() {
    // Initialize 10 28x28 matrices to store images.
    FloatMatrix** numbers = zeroMatrices(10, 28, 28);

    // Load all 10 images.
    numbers[0] = loadMatrix("data/0.bin", 28, 28);
    numbers[1] = loadMatrix("data/1.bin", 28, 28);
    numbers[2] = loadMatrix("data/2.bin", 28, 28);
    numbers[3] = loadMatrix("data/3.bin", 28, 28);
    numbers[4] = loadMatrix("data/4.bin", 28, 28);
    numbers[5] = loadMatrix("data/5.bin", 28, 28);
    numbers[6] = loadMatrix("data/6.bin", 28, 28);
    numbers[7] = loadMatrix("data/7.bin", 28, 28);
    numbers[8] = loadMatrix("data/8.bin", 28, 28);
    numbers[9] = loadMatrix("data/9.bin", 28, 28);

    // Display them as ASCII art.
    forEachMatrix(numbers, 10, displayMatrix);

    // Fre allocated memory.
    forEachMatrix(numbers, 10, freeMatrix);
}

void matrixMultiplicationTest() {
    srand(time(NULL));

    // Choose random matrix sizes.
    int m = 2 + rand() % 6;
    int n = 2 + rand() % 6;
    int p = 2 + rand() % 6;

    // Initialize some random matrices.
    FloatMatrix* mat1 = randomMatrix(m, n);
    FloatMatrix* mat2 = randomMatrix(n, p);

    // Initialize a matrix to store the result of the matrix multiplication.
    FloatMatrix* result = zeroMatrix(m, p);

    // Multiply matrix on the GPU and copy result to CPU.
    matrixMult(mat1, mat2, result);
    cudaDeviceSynchronize();
    copyFromDevice(result);

    // Print all the matrices.
    printMatrix(mat1);
    printMatrix(mat2);
    printMatrix(result);

    // Free allocated memory.
    freeMatrix(mat1);
    freeMatrix(mat2);
    freeMatrix(result);
}

void classifierTest() {
    FloatMatrix* number = loadMatrix("data/7.bin", 28, 28);
    FloatMatrix** inChannels = &number;

    ConvolutionLayer* conv1 = newConvolutionLayer(1, 6, 5, 5, 28, 28, TANH);
    loadConvolutionLayerParams(conv1, "data/conv1-weights.bin", "data/conv1-bias.bin");
    printf("Conv 1 :\n");
    displayConvolutionLayerKernals(conv1);

    AveragePoolingLayer* avgPool1 = newAveragePoolingLayer(6, 24, 24, 2);

    ConvolutionLayer* conv2 = newConvolutionLayer(6, 16, 5, 5, 12, 12, TANH);
    loadConvolutionLayerParams(conv2, "data/conv2-weights.bin", "data/conv2-bias.bin");
    printf("\nConv 2 :\n");
    displayConvolutionLayerKernals(conv2);

    AveragePoolingLayer* avgPool2 = newAveragePoolingLayer(16, 8, 8, 2);

    FlattenLayer* flatten = newFlattenLayer(16, 4, 4);

    DenseLayer* dense1 = newDenseLayer(16 * 4 * 4, 120, TANH);
    loadDenseLayerParams(dense1, "data/dense1-weights.bin", "data/dense1-bias.bin");
    // displaySignedMatrix(dense1->weights);

    DenseLayer* dense2 = newDenseLayer(120, 84, TANH);
    loadDenseLayerParams(dense2, "data/dense2-weights.bin", "data/dense2-bias.bin");
    // displaySignedMatrix(dense2->weights);

    DenseLayer* dense3 = newDenseLayer(84, 10, SOFTMAX);
    loadDenseLayerParams(dense3, "data/dense3-weights.bin", "data/dense3-bias.bin");
    // displaySignedMatrix(dense3->weights);


    evaluateConvolutionLayer(conv1, inChannels);
    evaluateAveragePoolingLayer(avgPool1, conv1->outChannels);
    evaluateConvolutionLayer(conv2, avgPool1->outChannels);
    evaluateAveragePoolingLayer(avgPool2, conv2->outChannels);
    evaluateFlattenLayer(flatten, avgPool2->outChannels);
    evaluateDenseLayer(dense1, flatten->output);
    evaluateDenseLayer(dense2, dense1->output);
    evaluateDenseLayer(dense3, dense2->output);

    cudaDeviceSynchronize();
    printf("\nConv 1 :\n");
    displayConvolutionLayerOutputs(conv1);

    printf("\nAverage Pooling 1 :\n");
    displayAveragePoolingOutputs(avgPool1);

    printf("\nConv 2 :\n");
    displayConvolutionLayerOutputs(conv2);

    printf("\nAverage Pooling 2 :\n");
    displayAveragePoolingOutputs(avgPool2);

    printf("\nFlatten Output :\n");
    copyFromDevice(flatten->output);
    displaySignedMatrix(flatten->output);

    printf("\nDense 1 Output :\n");
    copyFromDevice(dense1->output);
    displaySignedMatrix(dense1->output);

    printf("\nDense 2 Output :\n");
    copyFromDevice(dense2->output);
    displaySignedMatrix(dense2->output);

    printf("\nDense 3 Output :\n");
    copyFromDevice(dense3->output);
    printMatrix(dense3->output);
}

int main() {
    // blurDemo();
    // sobelDemo();
    // paramsReadTest();
    // imageReadTest();
    // matrixMultiplicationTest();

    classifierTest();
    
    return 0;
}