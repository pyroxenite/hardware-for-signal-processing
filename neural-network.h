#include "layer.h"
#include "matrix.h"

#ifndef __NEURAL_NETWORK__
#define __NEURAL_NETWORK__

/**
* Représente un réseau de neuronnes.
*/
typedef struct NeuralNetwork {
    /** Première couche du réseau. */
    Layer* firstLayer;

    /** Permet d'afficher les sorties des couches au moment de l'évaluation. */
    bool isVerbose;
} NeuralNetwork;

/**
* Initialise un réseau de neuronnes vide.
* @return Le réseau de neuronnes.
*/
__host__ NeuralNetwork* newNeuralNetwork();

/**
* Ajoute une couche à un réseau de neuronnes. Elle est insérée en dernière position.
* @param nn    Le réseau de neuronnes.
* @param layer La couche à ajouter.
*/
__host__ void addLayer(
    NeuralNetwork* nn, 
    Layer* layer
);

/**
* Active l'affichage des sorties des couches au moment de l'évaluation du réseau
* de neuronnes.
* @param nn Le réseau de neuronnes.
*/
__host__ void enableVerbose(
    NeuralNetwork* nn
);

/**
* Évalue toutes les couches du réseau de neuronnes et renvoie un pointeur vers la 
* sortie de la dernière couche évaluée.
* @param nn    Le réseau de neuronnes.
* @param input La matrice en entrée.
* @return      Le premier canal de la sortie de la dernière couche évaluée.
*/
__host__ FloatMatrix* forward(
    NeuralNetwork* nn, 
    FloatMatrix* input
);

/**
* Initalise un réseau convolutif permettant de classifier des chiffres manuscrits
* blancs sur un fond noir sur des images de taille 28x28.
* @param isVerbose  Permet d'afficher les sorties des couches au moment de 
*                   l'évaluation.
* @return           Le réseau convolutif.
*/
NeuralNetwork* newDigitClassifier();

#endif