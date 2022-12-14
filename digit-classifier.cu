#include "digit-classifier.h"

__host__ ConvolutionLayer* newConvolutionLayer(int channelCount, int kernalCount, int ker_m, int ker_n, int in_m, int in_n) {
    ConvolutionLayer* conv = (ConvolutionLayer*) malloc(sizeof(ConvolutionLayer));

    conv->channelCount = channelCount;
    conv->kernalCount = kernalCount;
    conv->kernals = zeroMatrices(kernalCount * channelCount, ker_m, ker_n);
    conv->bias = zeroMatrix(1, kernalCount);
    conv->outputs = zeroMatrices(kernalCount, in_m - ker_m + 1, in_n - ker_n + 1);

    return conv;
}

__host__ void loadConvolutionLayerParams(ConvolutionLayer* conv, const char* kernalsPath, const char* biasPath) {
    conv->kernals = loadMatrices(kernalsPath, conv->kernalCount * conv->channelCount, conv->kernals[0]->m, conv->kernals[0]->n);
    conv->bias = loadVector(biasPath, conv->kernalCount, ROW);
}

__host__ void displayConvolutionLayerKernals(ConvolutionLayer* conv) {
    forEachMatrix(conv->kernals, conv->kernalCount*conv->channelCount, displaySignedMatrix);
}

__host__ void displayConvolutionLayerOutputs(ConvolutionLayer* conv) {
    forEachMatrix(conv->outputs, conv->kernalCount, copyFromDevice);
    forEachMatrix(conv->outputs, conv->kernalCount, displaySignedMatrix);
}

__host__ void evaluateConvolutionLayer(ConvolutionLayer* conv, FloatMatrix** inputChannels) {
    
}