#include "neural-network.h"

__host__ NeuralNetwork* newNeuralNetwork() {
    NeuralNetwork* nn = (NeuralNetwork*) malloc(sizeof(NeuralNetwork));
    nn->firstLayer = NULL;
    nn->isVerbose = false;
    return nn;
}

__host__ void addLayer(NeuralNetwork* nn, Layer* layer) {
    if (nn->firstLayer == NULL) {
        nn->firstLayer = layer;
        return;
    }
    Layer* currentLayer = nn->firstLayer;
    while (currentLayer->nextLayer != NULL) {
        currentLayer = currentLayer->nextLayer;
    }
    currentLayer->nextLayer = layer;
}

__host__ void enableVerbose(NeuralNetwork* nn) {
    nn->isVerbose = true;
}

__host__ FloatMatrix* forward(NeuralNetwork* nn, FloatMatrix* input) {
    Layer* currentLayer = nn->firstLayer;
    FloatMatrix** data = &input;
    while (currentLayer != NULL) {
        evaluateLayer(currentLayer, data, nn->isVerbose);
        data = currentLayer->output;
        currentLayer = currentLayer->nextLayer;
    }
    cudaDeviceSynchronize();
    copyFromDevice(*data);
    return *data;
}

__host__ void freeNeuralNetwork(NeuralNetwork* nn) {
    Layer* currentLayer = nn->firstLayer;
    Layer* nextLayer;
    while (currentLayer->nextLayer != NULL) {
        nextLayer = currentLayer->nextLayer;
        freeLayer(currentLayer);
        currentLayer = nextLayer;
    }
    freeLayer(nextLayer);
    free(nn);
}

NeuralNetwork* newDigitClassifier() {
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