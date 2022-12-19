#include "unit-tests.h"

bool readMatrixTest() {
    FloatMatrix* matrix = loadMatrix("data/test.bin", 3, 3);

    float correctValues[] = { 
         0.0,  0.5,  1.0,
        -0.5,  0.0,  0.5,
        -1.0, -0.5,  0.0
    };

    for (int i=0; i<9; i++) {
        if (matrix->cpu[i] != correctValues[i]) {
            return false;
        }
    }

    freeMatrix(matrix);
    return true;
}

bool matrixMultiplicationTest() {
    srand(time(NULL));

    // Choose random matrix sizes.
    int m = 3;
    int n = 3;
    int p = 2;

    // Initialize some random matrices.
    FloatMatrix* mat1 = randomMatrix(m, n);
    FloatMatrix* mat2 = randomMatrix(n, p);

    // Initialize a matrix to store the result of the matrix multiplication.
    FloatMatrix* result = zeroMatrix(m, p);

    // Multiply matrix on the GPU and copy result to CPU.
    multiplyMatrices(mat1, mat2, result);
    cudaDeviceSynchronize();
    copyFromDevice(result);

    // Verify all coefficients.
    if (abs(result->cpu[0] - mat1->cpu[0]*mat2->cpu[0] + mat1->cpu[1]*mat2->cpu[2] + mat1->cpu[2]*mat2->cpu[4]) < FLOAT_TEST_EPSILON) 
        return false;
    if (abs(result->cpu[1] - mat1->cpu[0]*mat2->cpu[1] + mat1->cpu[1]*mat2->cpu[3] + mat1->cpu[2]*mat2->cpu[5]) < FLOAT_TEST_EPSILON) 
        return false;

    if (abs(result->cpu[2] - mat1->cpu[3]*mat2->cpu[0] + mat1->cpu[4]*mat2->cpu[2] + mat1->cpu[5]*mat2->cpu[4]) < FLOAT_TEST_EPSILON) 
        return false;
    if (abs(result->cpu[3] - mat1->cpu[3]*mat2->cpu[1] + mat1->cpu[4]*mat2->cpu[3] + mat1->cpu[5]*mat2->cpu[5]) < FLOAT_TEST_EPSILON) 
        return false;

    if (abs(result->cpu[4] - mat1->cpu[6]*mat2->cpu[0] + mat1->cpu[7]*mat2->cpu[2] + mat1->cpu[8]*mat2->cpu[4]) < FLOAT_TEST_EPSILON) 
        return false;
    if (abs(result->cpu[5] - mat1->cpu[6]*mat2->cpu[1] + mat1->cpu[7]*mat2->cpu[3] + mat1->cpu[8]*mat2->cpu[5]) < FLOAT_TEST_EPSILON) 
        return false;

    // Free allocated memory.
    freeMatrix(mat1);
    freeMatrix(mat2);
    freeMatrix(result);

    return true;
}


bool avgPoolTest() {
    FloatMatrix* mat1 = randomMatrix(4, 4);
    FloatMatrix* mat2 = zeroMatrix(2, 2);

    averagePool(mat1, mat2, 2);
    copyFromDevice(mat2);

    float sum;

    sum = 0;
    sum += mat1->cpu[0];
    sum += mat1->cpu[1];
    sum += mat1->cpu[4];
    sum += mat1->cpu[5];
    if (abs(sum/4 - mat2->cpu[0]) > FLOAT_TEST_EPSILON)
        return false;

    sum = 0;
    sum += mat1->cpu[2];
    sum += mat1->cpu[3];
    sum += mat1->cpu[6];
    sum += mat1->cpu[7];
    if (abs(sum/4 - mat2->cpu[1]) > FLOAT_TEST_EPSILON)
        return false;

    sum = 0;
    sum += mat1->cpu[8];
    sum += mat1->cpu[9];
    sum += mat1->cpu[12];
    sum += mat1->cpu[13];
    if (abs(sum/4 - mat2->cpu[2]) > FLOAT_TEST_EPSILON)
        return false;

    sum = 0;
    sum += mat1->cpu[10];
    sum += mat1->cpu[11];
    sum += mat1->cpu[14];
    sum += mat1->cpu[15];
    if (abs(sum/4 - mat2->cpu[3]) > FLOAT_TEST_EPSILON)
        return false;

    return true;
}

bool convolveTest() {
    FloatMatrix* matrix = randomMatrix(3, 3);
    FloatMatrix* kernal = randomMatrix(2, 2);
    FloatMatrix* result = zeroMatrix(2, 2);

    convolve(matrix, kernal, result);
    copyFromDevice(result);

    float val;
    
    val = 0;
    val += matrix->cpu[0]*kernal->cpu[0] + matrix->cpu[1]*kernal->cpu[1];
    val += matrix->cpu[3]*kernal->cpu[2] + matrix->cpu[4]*kernal->cpu[3];
    if (abs(result->cpu[0] - val) > FLOAT_TEST_EPSILON)
        return false;

    val = 0;
    val += matrix->cpu[1]*kernal->cpu[0] + matrix->cpu[2]*kernal->cpu[1];
    val += matrix->cpu[4]*kernal->cpu[2] + matrix->cpu[5]*kernal->cpu[3];
    if (abs(result->cpu[1] - val) > FLOAT_TEST_EPSILON)
        return false;

    val = 0;
    val += matrix->cpu[3]*kernal->cpu[0] + matrix->cpu[4]*kernal->cpu[1];
    val += matrix->cpu[6]*kernal->cpu[2] + matrix->cpu[7]*kernal->cpu[3];
    if (abs(result->cpu[2] - val) > FLOAT_TEST_EPSILON)
        return false;

    val = 0;
    val += matrix->cpu[4]*kernal->cpu[0] + matrix->cpu[5]*kernal->cpu[1];
    val += matrix->cpu[7]*kernal->cpu[2] + matrix->cpu[8]*kernal->cpu[3];
    if (abs(result->cpu[3] - val) > FLOAT_TEST_EPSILON)
        return false;

    return true;
}

bool flattenTest() {
    FloatMatrix** matrices = randomMatrices(2, 2, 2);
    FloatMatrix* flat = zeroMatrix(8, 1);

    flattenMatrices(matrices, 2, flat);
    copyFromDevice(flat);

    if (matrices[0]->cpu[0] != flat->cpu[0])
        return false;

    if (matrices[1]->cpu[0] != flat->cpu[1])
        return false;

    if (matrices[0]->cpu[1] != flat->cpu[2])
        return false;

    if (matrices[1]->cpu[1] != flat->cpu[3])
        return false;

    if (matrices[0]->cpu[2] != flat->cpu[4])
        return false;

    if (matrices[1]->cpu[2] != flat->cpu[5])
        return false;

    if (matrices[0]->cpu[3] != flat->cpu[6])
        return false;

    if (matrices[1]->cpu[3] != flat->cpu[7])
        return false;

    freeMatrices(matrices, 2);
    freeMatrix(flat);    

    return true;
}

bool classificationTest() {
    NeuralNetwork* cnn = newDigitClassifier();

    FloatMatrix* input;
    FloatMatrix* output;

    input = loadMatrix("data/0.bin", 28, 28);

    output = forward(cnn, input);
    if (argmax(output) != 0)
        return false;

    free(input);
    input = loadMatrix("data/1.bin", 28, 28);

    output = forward(cnn, input);
    if (argmax(output) != 1)
        return false;

    free(input);
    input = loadMatrix("data/2.bin", 28, 28);

    output = forward(cnn, input);
    if (argmax(output) != 2)
        return false;

    free(input);
    input = loadMatrix("data/3.bin", 28, 28);

    output = forward(cnn, input);
    if (argmax(output) != 3)
        return false;

    free(input);
    input = loadMatrix("data/4.bin", 28, 28);

    output = forward(cnn, input);
    if (argmax(output) != 4)
        return false;

    free(input);
    input = loadMatrix("data/5.bin", 28, 28);

    output = forward(cnn, input);
    if (argmax(output) != 5)
        return false;

    free(input);
    input = loadMatrix("data/6.bin", 28, 28);

    output = forward(cnn, input);
    if (argmax(output) != 6)
        return false;

    free(input);
    input = loadMatrix("data/7.bin", 28, 28);

    output = forward(cnn, input);
    if (argmax(output) != 7)
        return false;

    free(input);
    input = loadMatrix("data/8.bin", 28, 28);

    output = forward(cnn, input);
    if (argmax(output) != 8)
        return false;

    free(input);
    input = loadMatrix("data/9.bin", 28, 28);

    output = forward(cnn, input);
    if (argmax(output) != 9)
        return false;

    free(input);

    freeNeuralNetwork(cnn);

    return true;
}

void printUnitTestResult(const char* testName, bool success) {
    if (success) {
        printf("│ %-30s %s%sSUCCESS%s │\n", testName, BOLD, GREEN, RESET);
    } else {
        printf("│ %-30s %s%s FAILED%s │\n", testName, BOLD, RED, RESET);
    }
}

void runUnitTests() {
    printf("╭────────────────────────────────────────╮\n");
    printf("│ %sTest name                      Result%s  │\n", BOLD, RESET);
    printf("├────────────────────────────────────────┤\n");
    int successCount = 0;
    int testCount = 6;
    bool success;

    success = readMatrixTest();
    successCount += success;
    printUnitTestResult("Read matrix from file", success);

    success = convolveTest();
    successCount += success;
    printUnitTestResult("Convolution (GPU)", success);

    success = avgPoolTest();
    successCount += success;
    printUnitTestResult("Average pooling (GPU)", success);

    success = flattenTest();
    successCount += success;
    printUnitTestResult("Flatten (GPU)", success);

    success = matrixMultiplicationTest();
    successCount += success;
    printUnitTestResult("Matrix multiplication (GPU)", success);

    success = classificationTest();
    successCount += success;
    printUnitTestResult("Digit classification (0-9)", success);

    printf("├────────────────────────────────────────┤\n");
    printf("│ %sTotal                          %02d / %02d%s │\n", BOLD, successCount, testCount, RESET);
    printf("╰────────────────────────────────────────╯\n");
}