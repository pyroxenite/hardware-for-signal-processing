#include "digit-classifier.h"

__host__ NeuralNetwork* newNeuralNetwork() {
    NeuralNetwork* nn = (NeuralNetwork*) malloc(sizeof(NeuralNetwork));
    nn->firstLayer = NULL;
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

__host__ FloatMatrix** forward(NeuralNetwork* nn, FloatMatrix** input) {
    if (nn->firstLayer == NULL)
        return input; // Empty network, nothing to do.
    Layer* currentLayer = nn->firstLayer;
    FloatMatrix** data = input;
    while (currentLayer != NULL) {
        evaluateLayer(currentLayer, data);
        data = currentLayer->output;
        currentLayer = currentLayer->nextLayer;
    }
    cudaDeviceSynchronize();
    return data;
}

__host__ void evaluateLayer(Layer* layer, FloatMatrix** input) {
    if (layer->type == CONVOLUTION_LAYER) {
        ConvolutionLayer* convLayer = (ConvolutionLayer*) layer;
        evaluateConvolutionLayer(convLayer, input); 
    } else if (layer->type == AVERAGE_POOLING_LAYER) {
        AveragePoolingLayer* avgPoolLayer = (AveragePoolingLayer*) layer;
        evaluateAveragePoolingLayer(avgPoolLayer, input);
    } else if (layer->type == FLATTEN_LAYER) {
        FlattenLayer* flattenLayer = (FlattenLayer*) layer;
        evaluateFlattenLayer(flattenLayer, input);
    } else if (layer->type == DENSE_LAYER) {
        DenseLayer* denseLayer = (DenseLayer*) layer;
        evaluateDenseLayer(denseLayer, input);
    }
}

__host__ ConvolutionLayer* newConvolutionLayer(
    int inChannelCount, int outChannelCount, int ker_m, int ker_n, int in_m, int in_n, Activation activation
) {
    ConvolutionLayer* layer = (ConvolutionLayer*) malloc(sizeof(ConvolutionLayer));

    layer->type = CONVOLUTION_LAYER;
    layer->output = zeroMatrices(outChannelCount, in_m - ker_m + 1, in_n - ker_n + 1);
    layer->nextLayer = (Layer*) NULL;

    layer->inChannelCount = inChannelCount;
    layer->outChannelCount = outChannelCount;
    layer->kernals = zeroMatrices(outChannelCount * inChannelCount, ker_m, ker_n);
    layer->bias = zeroMatrix(1, outChannelCount);
    layer->activation = activation;

    return layer;
}

__host__ void loadConvolutionLayerParams(
    ConvolutionLayer* layer, const char* kernalsPath, const char* biasPath
) {
    forEachMatrix(layer->kernals, layer->inChannelCount * layer->outChannelCount, freeMatrix);
    layer->kernals = loadMatrices(
        kernalsPath, 
        layer->inChannelCount * layer->outChannelCount, 
        layer->kernals[0]->m, 
        layer->kernals[0]->n
    );
    freeMatrix(layer->bias);
    layer->bias = loadVector(
        biasPath, 
        layer->outChannelCount, 
        ROW
    );
}

__host__ void displayConvolutionLayerKernals(ConvolutionLayer* layer) {
    forEachMatrix(layer->kernals, layer->outChannelCount*layer->inChannelCount, displaySignedMatrix);
}

__host__ void evaluateConvolutionLayer(ConvolutionLayer* layer, FloatMatrix** input) {
    for (int out=0; out<layer->outChannelCount; out++) {
        setMatrixToZero(layer->output[out]);
        for (int in=0; in<layer->inChannelCount; in++) {
            convolve(
                input[in], 
                layer->kernals[in + out*layer->inChannelCount], 
                layer->output[out]
            );
        }
        addValueToMatrix(layer->output[out], layer->bias->cpu[out]);
        applyActivation(layer->output[out], layer->activation);
    }
}

__host__ void displayConvolutionLayerOutputs(ConvolutionLayer* layer) {
    forEachMatrix(layer->output, layer->outChannelCount, copyFromDevice);
    forEachMatrix(layer->output, layer->outChannelCount, displaySignedMatrix);
}

__host__ AveragePoolingLayer* newAveragePoolingLayer(int channelCount, int m, int n, int amount) {
    AveragePoolingLayer* layer = (AveragePoolingLayer*) malloc(sizeof(AveragePoolingLayer));

    layer->type = AVERAGE_POOLING_LAYER;
    layer->output = zeroMatrices(channelCount, m/amount, n/amount);
    layer->nextLayer = (Layer*) NULL;

    layer->channelCount = channelCount;
    layer->amount = amount;

    return layer;
}

__host__ void evaluateAveragePoolingLayer(AveragePoolingLayer* layer, FloatMatrix** input) {
    for (int i=0; i<layer->channelCount; i++) {
        averagePool(input[i], layer->output[i], layer->amount);
    }
}

__host__ void displayAveragePoolingOutputs(AveragePoolingLayer* layer) {
    forEachMatrix(layer->output, layer->channelCount, copyFromDevice);
    forEachMatrix(layer->output, layer->channelCount, displaySignedMatrix);
}


__host__ FlattenLayer* newFlattenLayer(int channelCount, int m, int n) {
    FlattenLayer* layer = (FlattenLayer*) malloc(sizeof(FlattenLayer));

    layer->type = FLATTEN_LAYER;
    layer->output = zeroMatrices(1, channelCount * m * n, 1);
    layer->nextLayer = (Layer*) NULL;

    layer->channelCount = channelCount;
    layer->m = m;
    layer->n = n;

    return layer;
}

__host__ void evaluateFlattenLayer(FlattenLayer* layer, FloatMatrix** input) {
    flattenMatrices(input, layer->channelCount, layer->output[0]);
}

__host__ DenseLayer* newDenseLayer(int inSize, int outSize, Activation activation) {
    DenseLayer* layer = (DenseLayer*) malloc(sizeof(DenseLayer));

    layer->type = DENSE_LAYER;
    layer->output = zeroMatrices(1, outSize, 1);
    layer->nextLayer = (Layer*) NULL;

    layer->weights = zeroMatrix(outSize, inSize);
    layer->bias = zeroMatrix(outSize, 1);
    layer->activation = activation;

    return layer;
}

__host__ void loadDenseLayerParams(DenseLayer* layer, const char* weightsPath, const char* biasPath) {
    freeMatrix(layer->weights);
    layer->weights = loadMatrix(weightsPath, layer->weights->m, layer->weights->n);
    freeMatrix(layer->bias);
    layer->bias = loadMatrix(biasPath, layer->bias->m, layer->bias->n);
}

__host__ void evaluateDenseLayer(DenseLayer* layer, FloatMatrix** input) {
    matrixMult(layer->weights, *input, *(layer->output));
    addMatrix(layer->bias, *(layer->output));
    applyActivation(*(layer->output), layer->activation);
}