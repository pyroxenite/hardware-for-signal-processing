#include "digit-classifier.h"

__host__ ConvolutionLayer* newConvolutionLayer(
    int inChannelCount, int outChannelCount, int ker_m, int ker_n, int in_m, int in_n, Activation activation
) {
    ConvolutionLayer* conv = (ConvolutionLayer*) malloc(sizeof(ConvolutionLayer));

    conv->inChannelCount = inChannelCount;
    conv->outChannelCount = outChannelCount;
    conv->kernals = zeroMatrices(outChannelCount * inChannelCount, ker_m, ker_n);
    conv->bias = zeroMatrix(1, outChannelCount);
    conv->outChannels = zeroMatrices(outChannelCount, in_m - ker_m + 1, in_n - ker_n + 1);
    conv->activation = activation;

    return conv;
}

__host__ void loadConvolutionLayerParams(
    ConvolutionLayer* conv, const char* kernalsPath, const char* biasPath
) {
    forEachMatrix(conv->kernals, conv->inChannelCount * conv->outChannelCount, freeMatrix);
    conv->kernals = loadMatrices(
        kernalsPath, 
        conv->inChannelCount * conv->outChannelCount, 
        conv->kernals[0]->m, 
        conv->kernals[0]->n
    );
    freeMatrix(conv->bias);
    conv->bias = loadVector(
        biasPath, 
        conv->outChannelCount, 
        ROW
    );
}

__host__ void displayConvolutionLayerKernals(ConvolutionLayer* conv) {
    forEachMatrix(conv->kernals, conv->outChannelCount*conv->inChannelCount, displaySignedMatrix);
}

__host__ void evaluateConvolutionLayer(ConvolutionLayer* conv, FloatMatrix** inChannels) {
    for (int out=0; out<conv->outChannelCount; out++) {
        setMatrixToZero(conv->outChannels[out]);
        for (int in=0; in<conv->inChannelCount; in++) {
            convolve(
                inChannels[in], 
                conv->kernals[in + out*conv->inChannelCount], 
                conv->outChannels[out]
            );
        }
        addValueToMatrix(conv->outChannels[out], conv->bias->cpu[out]);
        applyActivation(conv->outChannels[out], conv->activation);
    }
}

__host__ void displayConvolutionLayerOutputs(ConvolutionLayer* conv) {
    forEachMatrix(conv->outChannels, conv->outChannelCount, copyFromDevice);
    forEachMatrix(conv->outChannels, conv->outChannelCount, displaySignedMatrix);
}

__host__ AveragePoolingLayer* newAveragePoolingLayer(int channelCount, int m, int n, int amount) {
    AveragePoolingLayer* avgPool = (AveragePoolingLayer*) malloc(sizeof(AveragePoolingLayer));

    avgPool->channelCount = channelCount;
    avgPool->outChannels = zeroMatrices(channelCount, m/amount, n/amount);
    avgPool->amount = amount;

    return avgPool;
}

__host__ void evaluateAveragePoolingLayer(AveragePoolingLayer* avgPool, FloatMatrix** inChannels) {
    for (int i=0; i<avgPool->channelCount; i++) {
        averagePool(inChannels[i], avgPool->outChannels[i], avgPool->amount);
    }
}

__host__ void displayAveragePoolingOutputs(AveragePoolingLayer* avgPool) {
    forEachMatrix(avgPool->outChannels, avgPool->channelCount, copyFromDevice);
    forEachMatrix(avgPool->outChannels, avgPool->channelCount, displaySignedMatrix);
}


__host__ FlattenLayer* newFlattenLayer(int channelCount, int m, int n) {
    FlattenLayer* flatten = (FlattenLayer*) malloc(sizeof(FlattenLayer));

    flatten->channelCount = channelCount;
    flatten->m = m;
    flatten->n = n;
    flatten->output = zeroMatrix(channelCount * m * n, 1);

    return flatten;
}

__host__ void evaluateFlattenLayer(FlattenLayer* flatten, FloatMatrix** inChannels) {
    // for (int in=0; in<flatten->channelCount; in++) {
    //     for (int i=0; i<flatten->m*flatten->n; i++) {
    //         flatten->output->cpu[in * flatten->m * flatten->n + i] = inChannels[in]->cpu[i];
    //     }
    // }
    flattenMatrices(inChannels, flatten->channelCount, flatten->output);
}

__host__ DenseLayer* newDenseLayer(int inSize, int outSize, Activation activation) {
    DenseLayer* dense = (DenseLayer*) malloc(sizeof(DenseLayer));

    dense->weights = zeroMatrix(outSize, inSize);
    dense->bias = zeroMatrix(outSize, 1);
    dense->output = zeroMatrix(outSize, 1);

    return dense;
}

__host__ void loadDenseLayerParams(DenseLayer* dense, const char* weightsPath, const char* biasPath) {
    freeMatrix(dense->weights);
    dense->weights = loadMatrix(weightsPath, dense->weights->m, dense->weights->n);
    freeMatrix(dense->bias);
    dense->bias = loadMatrix(biasPath, dense->bias->m, dense->bias->n);
}

__host__ void evaluateDenseLayer(DenseLayer* dense, FloatMatrix* input) {
    matrixMult(dense->weights, input, dense->output);
    addMatrix(dense->bias, dense->output);
    applyActivation(dense->output, dense->activation);
}