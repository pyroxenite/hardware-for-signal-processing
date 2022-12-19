#include "demo.h"

void blurDemo() {
    int im_size = 31;
    int ker_size = 6;
    int res_size = im_size - ker_size + 1;

    // Initialize some matrices. (CPU & GPU)
    FloatMatrix* image = zeroMatrix(im_size, im_size);
    FloatMatrix* kernal = zeroMatrix(ker_size, ker_size);
    FloatMatrix* result = zeroMatrix(res_size, res_size);

    // Create example input image. (GPU)
    drawCircle(image, im_size/2.5, im_size/2.5, im_size/5.0, 0.4);
    drawCircle(image, 1.5*im_size/2.5, 1.5*im_size/2.5, im_size/5.0, 0.1);

    // Create example kernal. (GPU)
    drawCircle(kernal, ker_size/2 - 0.5, ker_size/2 - 0.5, ker_size/2, 0.1);

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
    displayMatrix(kernal);
    displayMatrix(result);

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

void kernalReadDemo() {
    // Read matrices from files.
    FloatMatrix** kernals = loadMatrices("data/conv1-weights.bin", 6, 5, 5);

    // Display them.
    forEachMatrix(kernals, 6, displaySignedMatrix);
    
    // Free allocated memory.
    forEachMatrix(kernals, 6, freeMatrix);
}

void imageReadDemo() {
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
    freeMatrices(numbers, 10);
}

void classificationDemo() {
    NeuralNetwork* cnn = newDigitClassifier();

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

    FloatMatrix* output;

    for (int i=0; i<10; i++) {
        displaySignedMatrix(numbers[i]);
        output = forward(cnn, numbers[i]);
        displayVectorAsBarGraph(output, 24, "Distribution de probabilité");
        printf("%sPrediction :%s %d\n\n", BOLD, RESET, argmax(output));
    }

    freeMatrices(numbers, 10);
    freeNeuralNetwork(cnn);
}