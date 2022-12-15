#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

typedef enum LayerType {
    CONVOLUTION_LAYER,
    AVERAGE_POOLING_LAYER,
    FLATTEN_LAYER,
    DENSE_LAYER
} LayerType;

typedef struct Layer {
    LayerType type;
    FloatMatrix** output;
    Layer* nextLayer;
} Layer;

typedef struct ConvolutionLayer {
    LayerType type;
    FloatMatrix** output;
    Layer* nextLayer;

    int inChannelCount;
    int outChannelCount;
    FloatMatrix** kernals;
    FloatMatrix* bias;
    Activation activation;
} ConvolutionLayer;

typedef struct AveragePoolingLayer {
    LayerType type;
    FloatMatrix** output;
    Layer* nextLayer;

    int channelCount;
    int amount;
} AveragePoolingLayer;

typedef struct FlattenLayer {
    LayerType type;
    FloatMatrix** output;
    Layer* nextLayer;

    int channelCount;
    int m;
    int n;
} FlattenLayer;

typedef struct DenseLayer {
    LayerType type;
    FloatMatrix** output;
    Layer* nextLayer;

    FloatMatrix* weights;
    FloatMatrix* bias;
    Activation activation;
} DenseLayer;

typedef struct NeuralNetwork {
    Layer* firstLayer;
} NeuralNetwork;

__host__ NeuralNetwork* newNeuralNetwork();

__host__ FloatMatrix** forward(
    NeuralNetwork* nn, 
    FloatMatrix** input
);

__host__ void addLayer(
    NeuralNetwork* nn, 
    Layer* layer
);

__host__ void evaluateLayer(
    Layer* layer, 
    FloatMatrix** input
);

__host__ ConvolutionLayer* newConvolutionLayer(
    int channelCount, 
    int kernalCount, 
    int ker_m, int ker_n, 
    int in_m,  int in_n,
    Activation activation
);

__host__ void loadConvolutionLayerParams(
    ConvolutionLayer* conv, 
    const char* kernalsPath, 
    const char* biasPath
);

__host__ void displayConvolutionLayerKernals(
    ConvolutionLayer* conv
);

__host__ void displayConvolutionLayerOutputs(
    ConvolutionLayer* conv
);

__host__ void evaluateConvolutionLayer(
    ConvolutionLayer* conv, 
    FloatMatrix** input
);


__host__ AveragePoolingLayer* newAveragePoolingLayer(
    int channelCount, 
    int m, int n,
    int amount
);

__host__ void displayAveragePoolingOutputs(
    AveragePoolingLayer* avgPool
);

__host__ void evaluateAveragePoolingLayer(
    AveragePoolingLayer* avgPool, 
    FloatMatrix** input
);


__host__ FlattenLayer* newFlattenLayer(
    int channelCount, 
    int m, int n
);

__host__ void evaluateFlattenLayer(
    FlattenLayer* flatten, 
    FloatMatrix** input
);


__host__ DenseLayer* newDenseLayer(
    int inSize, 
    int outSize,
    Activation activation
);

__host__ void loadDenseLayerParams(
    DenseLayer* dense, 
    const char* weightsPath, 
    const char* biasPath
);

__host__ void evaluateDenseLayer(
    DenseLayer* dense, 
    FloatMatrix** input
);


// __host__ DigitClassifier* newDigitClassifier();

// __host__ void loadDigitClassifierParams(
//     DigitClassifier* classif, 
//     const char* dir
// );

// __host__ int evaluateDigitClassifier(
//     DigitClassifier* classif, 
//     FloatMatrix* input
// );