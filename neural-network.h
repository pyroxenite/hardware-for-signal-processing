#include "layer.h"
#include "matrix.h"

#ifndef __NEURAL_NETWORK__
#define __NEURAL_NETWORK__

typedef struct NeuralNetwork {
    Layer* firstLayer;
    bool isVerbose;
} NeuralNetwork;

__host__ NeuralNetwork* newNeuralNetwork();

__host__ void addLayer(
    NeuralNetwork* nn, 
    Layer* layer
);

__host__ void enableVerbose(
    NeuralNetwork* nn
);

__host__ FloatMatrix* forward(
    NeuralNetwork* nn, 
    FloatMatrix* input
);

#endif