#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matrix.h"
#include "layer.h"
#include "neural-network.h"

//#include "tests.cu"
//#include "demos.cu"

NeuralNetwork* newClassifier() {
    ConvolutionLayer* conv1 = newConvolutionLayer(1, 6, 5, 5, 28, 28, TANH);
    loadConvolutionLayerParams(conv1, "data/conv1-weights.bin", "data/conv1-bias.bin");

    ConvolutionLayer* conv2 = newConvolutionLayer(6, 16, 5, 5, 12, 12, TANH);
    loadConvolutionLayerParams(conv2, "data/conv2-weights.bin", "data/conv2-bias.bin");

    DenseLayer* dense1 = newDenseLayer(16 * 4 * 4, 120, TANH);
    loadDenseLayerParams(dense1, "data/dense1-weights.bin", "data/dense1-bias.bin");

    DenseLayer* dense2 = newDenseLayer(120, 84, TANH);
    loadDenseLayerParams(dense2, "data/dense2-weights.bin", "data/dense2-bias.bin");

    DenseLayer* dense3 = newDenseLayer(84, 10, SOFTMAX);
    loadDenseLayerParams(dense3, "data/dense3-weights.bin", "data/dense3-bias.bin");

    NeuralNetwork* cnn = newNeuralNetwork();
    // enableVerbose(cnn);

    addLayer(cnn, (Layer*) conv1);
    addLayer(cnn, (Layer*) newAveragePoolingLayer(6, 24, 24, 2));
    addLayer(cnn, (Layer*) conv2);
    addLayer(cnn, (Layer*) newAveragePoolingLayer(16, 8, 8, 2));
    addLayer(cnn, (Layer*) newFlattenLayer(16, 4, 4));
    addLayer(cnn, (Layer*) dense1);
    addLayer(cnn, (Layer*) dense2);
    addLayer(cnn, (Layer*) dense3);

    return cnn;
}
 
int main() {
    // blurDemo();
    // sobelDemo();
    
    // paramsReadTest();
    // imageReadTest();
    // convolveTest();
    // biasAndActivationTest();
    // avgPoolTest();
    // flattenTest();
    // matrixMultiplicationTest();
    // softMaxTest();

    NeuralNetwork* cnn = newClassifier();

    FloatMatrix* input;
    FloatMatrix* output;

    input = loadMatrix("data/0.bin", 28, 28);
    displayMatrix(input);

    output = forward(cnn, input);
    displayVectorAsBarGraph(output, 24, "Prédiction");

    freeMatrix(input);
    input = loadMatrix("data/1.bin", 28, 28);
    displayMatrix(input);

    output = forward(cnn, input);
    displayVectorAsBarGraph(output, 24, "Prédiction");

    freeMatrix(input);
    input = loadMatrix("data/2.bin", 28, 28);
    displayMatrix(input);

    output = forward(cnn, input);
    displayVectorAsBarGraph(output, 24, "Prédiction");

    freeMatrix(input);
    input = loadMatrix("data/3.bin", 28, 28);
    displayMatrix(input);

    output = forward(cnn, input);
    displayVectorAsBarGraph(output, 24, "Prédiction");

    freeMatrix(input);
    input = loadMatrix("data/4.bin", 28, 28);
    displayMatrix(input);

    output = forward(cnn, input);
    displayVectorAsBarGraph(output, 24, "Prédiction");

    return 0;
}