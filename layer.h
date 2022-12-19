#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

#ifndef __LAYER__
#define __LAYER__

/**
* Énumération des différents types de couches. Utilisé pour 
* pour distnguer les couches une fois castées en couche génériques.
*/
typedef enum LayerType {
    CONVOLUTION_LAYER,
    AVERAGE_POOLING_LAYER,
    FLATTEN_LAYER,
    DENSE_LAYER
} LayerType;

/**
* Type générique pour représenter une couche d'un réseau de neuronnes.
* Les types suivants reprennent les trois premiers champs de sorte 
* qu'une couche spécifique puisse être castée en couche générique.
* Par exemple, étant donné une `(ConvolutionLayer) convLayer` on peut 
* faire `(Layer*) convLayer` en présevant l'accès aux champs de base,
* définis ici.
*
* C'est en quelque sorte la "classe mère" pour les couches.
*/
typedef struct Layer {
    /** Définit comment traiter cette couche. */
    LayerType type; 

    /** Sortie de la couche. C'est un tableau de matrices pour 
    * pouvoir gérer les cas où plusieurs channels sont utilisées. */
    FloatMatrix** output;

    /** Pointeur vers la couche suivante. */
    Layer* nextLayer; 
} Layer;

/**
* Couche convolutive.
*/
typedef struct ConvolutionLayer {
    LayerType type;
    FloatMatrix** output;
    Layer* nextLayer;

    /** Nombre de canaux en entrée. */
    int inChannelCount;

    /** Nombre de canaux en sortie. */
    int outChannelCount;

    /** Tableau des noyaux de convolution. */
    FloatMatrix** kernals;

    /** Biais devant être appliqué après la convolution. */
    FloatMatrix* bias;

    /** Choix de la fonction d'activation. (`TANH` ou `SOFTMAX`) */
    Activation activation;
} ConvolutionLayer;

/**
* Couche de sous-échantillonnage par moyennage.
*/
typedef struct AveragePoolingLayer {
    LayerType type;
    FloatMatrix** output;
    Layer* nextLayer;

    /** Nombre de canaux en entrée. */
    int channelCount;

    /** Facteur de sous-échantillonnage. `2` pour diviser les 
    * dimensions de l'image par deux. */
    int amount;
} AveragePoolingLayer;

/** 
* Couche de verticalisation. Elle permettant de passer de $n$ canaux
* en entrée (3D) à un unique vecteur en sortie (1D).
*/
typedef struct FlattenLayer {
    LayerType type;
    FloatMatrix** output;
    Layer* nextLayer;

    /** Nombre de canaux en entrée. */
    int channelCount;

    /** Nombre de lignes dans les canaux. */
    int m;

    /** Nombre de colonnes dans les canaux. */
    int n;
} FlattenLayer;

/** 
* Couche dense. Tous les neurones sont connectés à toutes les entrées. 
*/
typedef struct DenseLayer {
    LayerType type;
    FloatMatrix** output;
    Layer* nextLayer;

    /** Matrice des poids de dimension (outSize, inSize). */
    FloatMatrix* weights;

    /** Biais. */
    FloatMatrix* bias;

    /** Choix de la fonction d'activation. (`TANH` ou `SOFTMAX`) */
    Activation activation;
} DenseLayer;

/**
* Évalue une couche générique en fonction de son champ `type`. 
* @param layer   La couche à évaluer.
* @param input   L'entrée sous la forme d'un tableau de matrices.
* @param verbose Permet d'afficher la sortie de la couche après 
*                évaluation.
*/
__host__ void evaluateLayer(
    Layer* layer,
    FloatMatrix** input,
    bool verbose
);

/**
* Initialise une couche convolutive.
* @param inChannelCount  Nombre de canaux en entrée.
* @param outChannelCount Nombre de canaux en sortie.
* @param ker_m           Nombre de lignes par noyau de convolution.
* @param ker_n           Nombre de colonnes par noyau de convolution.
* @param in_m            Nombre de lignes par canal en entrée.
* @param in_n            Nombre de colonnes par canal en entrée.
* @param activation      Choix de la fonction d'activation.
* @return                Pointeur vers la couche initialisée.
*/
__host__ ConvolutionLayer* newConvolutionLayer(
    int inChannelCount, 
    int outChannelCount, 
    int ker_m, int ker_n, 
    int in_m,  int in_n,
    Activation activation
);

/**
* Charge les poids et biais d'une couche convolutive à partir de deux 
* fichiers binaires. Les poids doivent respecter le format suivant : 
*  - virgule flottante sur 32 bits ;
*  - dimensions : (outChannelCount, inChannelCount, ker_m, ker_n).
* Les biais sont également en virgule flottante sur 32 bits. 
* @param layer       La couche convolutive dont on charge les 
                     paramètres.
* @param kernalsPath Chemin vers le fichier binaire content les noyaux.
* @param biasPath    Chemin vers le fichier binaire content les biais.
*/
__host__ void loadConvolutionLayerParams(
    ConvolutionLayer* conv, 
    const char* kernalsPath, 
    const char* biasPath
);

/** 
* Affiche les noyaux de convolution d'une couche convolutive.
* @params conv La couche convolutive.
*/
__host__ void displayConvolutionLayerKernals(
    ConvolutionLayer* conv
);

/** 
* Affiche la sortie d'une couche convolutive.
* @params conv La couche convolutive.
*/
__host__ void displayConvolutionLayerOutputs(
    ConvolutionLayer* conv
);

/**
* @param layer   La couche à évaluer.
* @param input   L'entrée sous la forme d'un tableau de matrices.
* @param verbose Permet d'afficher la sortie de la couche après 
*                évaluation.
*/
__host__ void evaluateConvolutionLayer(
    ConvolutionLayer* conv, 
    FloatMatrix** input,
    bool verbose
);

/**
* Initialise une couche de sous-échantillonnage par moyennage.
* @params channelCount Nombre de canaux.
* @params m            Nombre de lignes par canal en entrée.
* @params n            Nombre de colonnes par canal en entrée.
* @params amount       Facteur de sous-échantillonnage.
* @return              Pointeur vers la couche initialisée.
*/
__host__ AveragePoolingLayer* newAveragePoolingLayer(
    int channelCount, 
    int m, int n,
    int amount
);

/**
* Affiche la sortie d'une couche de sous-échantillonnage par moyennage.
* @params avgPool La couche en question.
*/
__host__ void displayAveragePoolingOutputs(
    AveragePoolingLayer* avgPool
);

/**
* @param layer   La couche à évaluer.
* @param input   L'entrée sous la forme d'un tableau de matrices.
* @param verbose Permet d'afficher la sortie de la couche après 
*                évaluation.
*/
__host__ void evaluateAveragePoolingLayer(
    AveragePoolingLayer* layer, 
    FloatMatrix** input,
    bool verbose
);

/**
* Initialise une couche de verticalisation.
* @params channelCount Nombre de canaux.
* @params m            Nombre de lignes par canal en entrée.
* @params n            Nombre de colonnes par canal en entrée.
* @return              Pointeur vers la couche initialisée.
*/
__host__ FlattenLayer* newFlattenLayer(
    int channelCount, 
    int m, int n
);

/**
* Évalue une couche 
* @param layer   La couche à évaluer.
* @param input   L'entrée sous la forme d'un tableau de matrices.
* @param verbose Permet d'afficher la sortie de la couche après 
*                évaluation.
*/
__host__ void evaluateFlattenLayer(
    FlattenLayer* layer, 
    FloatMatrix** input,
    bool verbose
);

/**
* Initialise une couche dense.
* @params inSize       Taille du vecteur en entrée.
* @params outSize      Taille du vecteur en sortie.
* @return              Pointeur vers la couche initialisée.
*/
__host__ DenseLayer* newDenseLayer(
    int inSize, 
    int outSize,
    Activation activation
);

/**
* Charge les poids et biais d'une couche dense à partir de deux 
* fichiers binaires. Les poids doivent respecter le format suivant : 
*  - virgule flottante sur 32 bits ;
*  - dimensions : (outSize, inSize).
* Les biais sont également en virgule flottante sur 32 bits. 
* @param layer       La couche dense dont on charge les paramètres.
* @param kernalsPath Chemin vers le fichier binaire content les noyaux.
* @param biasPath    Chemin vers le fichier binaire content les biais.
*/
__host__ void loadDenseLayerParams(
    DenseLayer* dense, 
    const char* weightsPath, 
    const char* biasPath
);

/**
* Évalue une couche dense.
* @param layer   La couche à évaluer.
* @param input   L'entrée sous la forme d'un tableau de matrices.
* @param verbose Permet d'afficher la sortie de la couche après 
*                évaluation.
*/
__host__ void evaluateDenseLayer(
    DenseLayer* dense, 
    FloatMatrix** input,
    bool verbose
);

#endif