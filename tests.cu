#include "matrix.h"

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


void avgPoolTest() {
    FloatMatrix* mat1 = randomMatrix(4, 4);
    FloatMatrix* mat2 = zeroMatrix(2, 2);

    averagePool(mat1, mat2, 2);
    copyFromDevice(mat2);

    printMatrix(mat1);
    printMatrix(mat2);

    float sum = 0;
    sum += mat1->cpu[8];
    sum += mat1->cpu[9];
    sum += mat1->cpu[12];
    sum += mat1->cpu[13];
    printf("%f\n", sum/4);
}

void convolveTest() {
    FloatMatrix* matrix = randomMatrix(10, 10);
    FloatMatrix* kernal = randomMatrix(2, 2);
    FloatMatrix* result = zeroMatrix(9, 9);

    convolve(matrix, kernal, result);
    copyFromDevice(result);

    printMatrix(matrix);
    printMatrix(kernal);
    printMatrix(result);
}

void biasAndActivationTest() {
    FloatMatrix* matrix = randomMatrix(10, 10);
    printMatrix(matrix);

    addValueToMatrix(matrix, 1);
    copyFromDevice(matrix);

    printMatrix(matrix);

    applyActivation(matrix, TANH);
    copyFromDevice(matrix);

    printMatrix(matrix);
}

void flattenTest() {
    FloatMatrix** matrices = randomMatrices(3, 2, 2);
    FloatMatrix* flat = zeroMatrix(12, 1);

    flattenMatrices(matrices, 3, flat);
    copyFromDevice(flat);

    forEachMatrix(matrices, 3, printMatrix);
    printMatrix(flat);
}

void softMaxTest() {
    FloatMatrix* vector = randomMatrix(1, 3);

    printMatrix(vector);

    applyActivation(vector, SOFTMAX);
    copyFromDevice(vector);

    printMatrix(vector);
}