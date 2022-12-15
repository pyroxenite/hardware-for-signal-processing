#include "digit-classifier.h"

__host__ ConvolutionLayer* newConvolutionLayer(
    int inChannelCount, int outChannelCount, int ker_m, int ker_n, int in_m, int in_n
) {
    ConvolutionLayer* conv = (ConvolutionLayer*) malloc(sizeof(ConvolutionLayer));

    conv->inChannelCount = inChannelCount;
    conv->outChannelCount = outChannelCount;
    conv->kernals = zeroMatrices(outChannelCount * inChannelCount, ker_m, ker_n);
    conv->bias = zeroMatrix(1, outChannelCount);
    conv->outChannels = zeroMatrices(outChannelCount, in_m - ker_m + 1, in_n - ker_n + 1);

    return conv;
}

__host__ void loadConvolutionLayerParams(
    ConvolutionLayer* conv, const char* kernalsPath, const char* biasPath
) {
    conv->kernals = loadMatrices(
        kernalsPath, 
        conv->inChannelCount * conv->outChannelCount, 
        conv->kernals[0]->m, 
        conv->kernals[0]->n
    );
    conv->bias = loadVector(
        biasPath, 
        conv->outChannelCount, 
        ROW
    );
}

__host__ void displayConvolutionLayerKernals(ConvolutionLayer* conv) {
    forEachMatrix(conv->kernals, conv->outChannelCount*conv->inChannelCount, displaySignedMatrix);
}

__host__ void displayConvolutionLayerOutputs(ConvolutionLayer* conv) {
    forEachMatrix(conv->outChannels, conv->outChannelCount, copyFromDevice);
    forEachMatrix(conv->outChannels, conv->outChannelCount, displaySignedMatrix);
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
        applyActivation(conv->outChannels[out]);
    }
}

__host__ AveragePoolingLayer* newAveragePoolingLayer(int channelCount, int m, int n, int amount) {
    AveragePoolingLayer* avgPool = (AveragePoolingLayer*) malloc(sizeof(AveragePoolingLayer));

    avgPool->channelCount = channelCount;
    avgPool->outChannels = zeroMatrices(channelCount, m/amount, n/amount);
    avgPool->amount = amount;

    return avgPool;
}

__host__ void displayAveragePoolingOutputs(AveragePoolingLayer* avgPool) {
    forEachMatrix(avgPool->outChannels[i], avg->channelCount, displaySignedMatrix);
}

__host__ void evaluateAveragePoolingLayer(AveragePoolingLayer* avgPool, FloatMatrix** inChannels) {
    for (int i=0; i<avg->channelCount; i++) {
        averagePool(inChannels[i], avgPool->outChannels[i], avgPool->amount);
    }
}