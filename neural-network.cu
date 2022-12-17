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
    if (nn->firstLayer == NULL)
        return input; // Empty network, nothing to do.
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