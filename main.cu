#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "matrix.h"
#include "layer.h"
#include "neural-network.h"
#include "unit-tests.h"
#include "demo.h"

void printHelp() {
    printf("Les commandes suivantes sont disponibles :\n");
    printf(" * classify <path> [--verbose | --bar-graph]\n");
    printf(" * test\n");
    printf(" * demo <demo-name>\n");
    printf("    * classify-all\n");
    printf("    * blur\n");
    printf("    * sobel\n");
    printf("    * kernal-read\n");
    printf("    * image-read\n");
}

int main(int argc, char** argv) {

    if (argc == 1) {
        printHelp();
        return 0;
    }

    if (argc == 2 && strcmp(argv[1], "test") == 0) {
        runUnitTests();
        return 0;
    }

    if (argc == 3 && strcmp(argv[1], "demo") == 0) {
        if (strcmp(argv[2], "blur") == 0) {
            blurDemo();
            return 0;
        }

        if (strcmp(argv[2], "sobel") == 0) {
            sobelDemo();
            return 0;
        }

        if (strcmp(argv[2], "kernal-read") == 0) {
            kernalReadDemo();
            return 0;
        }

        if (strcmp(argv[2], "image-read") == 0) {
            imageReadDemo();
            return 0;
        }  

        if (strcmp(argv[2], "classify-all") == 0) {
            classificationDemo();
            return 0;
        }
    }

    if (argc == 3 && strcmp(argv[1], "display") == 0) {
        const char* path = argv[2];
        FloatMatrix* input = loadMatrix(path, 28, 28);

        displayMatrix(input);

        free(input);
        return 0;
    }

    if ((argc == 3 || argc == 4) && strcmp(argv[1], "classify") == 0) {
        const char* path = argv[2];
        NeuralNetwork* cnn = newDigitClassifier();

        if (argc == 4 && (strcmp(argv[3], "--verbose") == 0 || strcmp(argv[3], "-v") == 0)) {
            enableVerbose(cnn);
        }

        FloatMatrix* input = loadMatrix(path, 28, 28);
        FloatMatrix* output = forward(cnn, input);

        if (argc == 4 && (strcmp(argv[3], "--bar-graph") == 0)) {
            displayVectorAsBarGraph(output, 24, "Distibution de probabilité");
        } else {
            printf("%d\n", argmax(output));
        }

        freeMatrix(input);
        freeNeuralNetwork(cnn);

        return 0;
    }

    printf("Commande non reconnue. ");
    printHelp();
    return 0;
}