#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

typedef struct ConvolutionLayer {
    int inChannelCount;
    int outChannelCount;
    FloatMatrix** kernals;
    FloatMatrix* bias;
    FloatMatrix** outChannels;
    Activation activation;
} ConvolutionLayer;


typedef struct AveragePoolingLayer {
    int channelCount;
    FloatMatrix** outChannels;
    int amount;
} AveragePoolingLayer;


typedef struct FlattenLayer {
    int channelCount;
    int m;
    int n;
    FloatMatrix* output;
} FlattenLayer;


typedef struct DenseLayer {
    FloatMatrix* weights;
    FloatMatrix* bias;
    FloatMatrix* output;
    Activation activation;
} DenseLayer;


typedef struct DigitClassifier {
    FloatMatrix* input;
    ConvolutionLayer* conv1;
    AveragePoolingLayer* avgPool1;
    ConvolutionLayer* conv2;
    AveragePoolingLayer* avgPool2;
    FlattenLayer* flatten;
    DenseLayer* dense1;
    DenseLayer* dense2;
    DenseLayer* dense3;
} DigitClassifier;


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
    FloatMatrix** inputChannels
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
    FloatMatrix** inChannels
);


__host__ FlattenLayer* newFlattenLayer(
    int channelCount, 
    int m, int n
);

__host__ void evaluateFlattenLayer(
    FlattenLayer* flatten, 
    FloatMatrix** inChannels
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
    FloatMatrix* input
);


__host__ DigitClassifier* newDigitClassifier();

__host__ void loadDigitClassifierParams(
    DigitClassifier* classif, 
    const char* dir
);

__host__ int evaluateDigitClassifier(
    DigitClassifier* classif, 
    FloatMatrix* input
);