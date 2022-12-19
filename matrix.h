#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <string.h>
#include "terminal-colors.h"

#ifndef __MATRIX__
#define __MATRIX__

#define COLUMN 1
#define ROW 0

/**
* Représente un choix de fonction d'activation pour une couche.
*/
typedef enum Activation { 
    TANH = 1, 
    SOFTMAX = 2 
} Activation;

/**
* Représnte une matrice.
*/
typedef struct FloatMatrix {
    /** Pointeur vers le tableau dans la mémoire du CPU. */
    float* cpu;

    /** Pointeur vers le tableau dans la mémoire du GPU. */
    float* gpu;

    /** Nombre de lignes. */
    int m;

    /** Nombre de colonnes. */
    int n;
} FloatMatrix;

/////////////////////////////// Initializers ///////////////////////////////

/**
* Initialise une matrice à partir d'un tableau coté CPU.
* Copie les valeurs vers le GPU à la même occasion.
* @param cpu Tableau coté CPU.
* @param m   Nombre de lignes.
* @param n   Nombre de colonnes.
* @return    La matrice initialisée.
*/
__host__ FloatMatrix* newMatrix(float* cpu, int m, int n);

/**
* Nouvelle matrice nulle.
* @param m Nombre de lignes.
* @param n Nombre de colonnes.
* @return  La matrice initialisée.
*/
__host__ FloatMatrix* zeroMatrix(int m, int n);

/**
* Tableau de matrices nulles.
* @param count Nombre matrices.
* @param m     Nombre de lignes.
* @param n     Nombre de colonnes.
* @return      Les matrices initialisées.
*/
__host__ FloatMatrix** zeroMatrices(int count, int m, int n);

/**
* Nouvelle matrice aléatoire avec des coefficients entre 0 et 1.
* @param m Nombre de lignes.
* @param n Nombre de colonnes.
* @return  La matrice initialisée.
*/
__host__ FloatMatrix* randomMatrix(int n, int m);

/**
* Tableau de matrices aléatoires.
* @param count Nombre matrices.
* @param m     Nombre de lignes.
* @param n     Nombre de colonnes.
* @return      Les matrices initialisées.
*/
__host__ FloatMatrix** randomMatrices(int count, int n, int m);

/**
* Charge une matrice à partir d'un fichier. Les coefficients doivent être
* en virgule flottante sur 32 bits.
* @param filename Chemin vers le fichier contenant les coefficients.
* @param m        Nombre de lignes.
* @param n        Nombre de colonnes.
* @return         La matrice fraichement initialisée.
*/
__host__ FloatMatrix* loadMatrix(const char* filename, int m, int n);

/**
* Charge un vecteur à partir d'un fichier. Les coefficients doivent être 
* en virgule flottante sur 32 bits.
* @param filename Chemin vers le fichier contenant les coefficients.
* @param n        Nombre de coefficients.
* @param isColumn Orientation.
* @return         La matrice fraichement initialisée.
*/
__host__ FloatMatrix* loadVector(const char* filename, int n, int isColumn);

/**
* Charge plusieurs matrices venant d'un même fichier. Les coefficients 
* doivent être en virgule flottante sur 32 bits. 
* @param filename Chemin vers le fichier contenant les coefficients.
* @param count    Nombre de matrices.
* @param m        Nombre de lignes.
* @param n        Nombre de colonnes.
* @return         Les matrices fraichement initialisées.
*/
__host__ FloatMatrix** loadMatrices(const char* filename, int count, int m, int n);

///////////////////////////// Memory management /////////////////////////////

/**
* Copie la matrice vers le GPU.
* @param matrix La matrice en question.
*/
__host__ void copyToDevice(FloatMatrix* matrix);

/**
* Copie la matrice vers le CPU.
* @param matrix La matrice en question.
*/
__host__ void copyFromDevice(FloatMatrix* matrix);

/**
* Libère la mémoire allouée à la matrice.
* @param matrix Matrice dont on libère la mémoire.
*/
__host__ void freeMatrix(FloatMatrix* matrix);

/**
* Libère la mémoire allouée aux matrices d'un tableau ainsi que celle 
* allouée au tableau de matrices.
* @param matrices Tableau de matrices.
* @param count    Taille du tableau.
*/
__host__ void freeMatrices(FloatMatrix** matrices, int count);

///////////////////////////////// Utilities /////////////////////////////////

/**
* Évalue une fonction pour chaque matrice d'un tableau de matrices.
* @param matrices Tableau de matrices.
* @param count    Nombre de matrices.
* @param fun      Pointeur vers une fonction qui prend une matrice en 
*                 entrée et ne renvoie rien. Par exemple, les fonctions 
*                 suivantes.
*/
__host__ void forEachMatrix(FloatMatrix** matrices, int count, void (*fun)(FloatMatrix* matrix));

/////////////////////////////////// Maths ///////////////////////////////////

/**
* Met tous les coefficients d'une matrice à zéro. La fonction opère du 
* côté GPU.
* @param matrix La matrice en question.
*/
__host__ void setMatrixToZero(FloatMatrix* matrix);

/**
* Ajoute une matrice à une autre. La fonction opère du côté GPU.
* @param matrix La matrice à ajouter.
* @param sum    La matrice à modifier.
*/
__host__ void addMatrix(FloatMatrix* matrix, FloatMatrix* sum);

/**
* Ajoute une valeur à tous les coefficents d'une matrice. La fonction 
* opère du côté GPU.
* @param matrix La matrice à modifier.
* @param sum    La valeur à ajouter.
*/
__host__ void addValueToMatrix(FloatMatrix* matrix, float value);

/**
* Multiplie une matrice par un scalaire. La fonction opère du côté GPU.
* @param matrix La matrice à modifier.
* @param sum    Le facteur multiplicatif.
*/
__host__ void scaleMatrix(FloatMatrix* matrix, float value);

/**
* Convolue une matrice avec une autre. La fonction opère du côté GPU.
* @param image  Image à convoluer. 
* @param kernal Noyau de convolution. 
* @return       Résultat de la convolution.
*/
__host__ void convolve(FloatMatrix* image, FloatMatrix* kernal, FloatMatrix* result);

/**
* Dessine un cercle sur une matrice. Permet d'effectuer des essais. La 
* fonction opère du côté GPU.
* @param matrix La matrice sur laquelle dessiner.
* @param x      Position X du centre du cercle.
* @param y      Position Y du centre du cercle.
* @param r      Rayon du cercle.
*/
__host__ void drawCircle(FloatMatrix* matrix, float x, float y, float r, float color);

/**
* Effectue un sous-échantillonnage par moyennage. La fonction opère du
* côté GPU.
* @param input  Matrice à sous-échantilloner.
* @param output Résultat du sous-échantillonage.
* @param amount Facteur de sous-échantillonage.
*/
__host__ void averagePool(FloatMatrix* input, FloatMatrix* output, int amount);

/**
* Applique une fonction d'activation à chaque coefficient d'une matrice. 
* La fonction opère du côté GPU.
* @param matrix     La matrice à laquelle on applique la fonction 
*                   d'activation.
* @param activation La fonction d'activation.
*/
__host__ void applyActivation(FloatMatrix* matrix, Activation activation);

/**
* Transforme un tableau de matrices en un unique vecteur colonne. La 
* fonction opère du côté GPU. La manière dont le tableau est applati est 
* contre intuitif: la boucle la plus profonde est celle correspondant aux
* canaux. Ainsi, pour le pixel $(i, j)$, on parcourt tous les 
* coefficients par canal avant de modifier $i$ ou $j$. Voir le test 
* unitaire associé pour un exemple.
* @param matrices Tableau de de matrices.
* @param count    Longeure du tableau de matrices.
* @param output   Vecteur en sortie.
*/
__host__ void flattenMatrices(FloatMatrix** matrices, int count, FloatMatrix* output);

/**
* Multiplie deux matrices. La fonction opère du côté GPU.
* @param mat1   La première matrice.
* @param mat2   La deuxième matrice.
* @param result Le résulat du produit.
*/
__host__ void multiplyMatrices(FloatMatrix* mat1, FloatMatrix* mat2, FloatMatrix* result);

/**
* Renvoie l'indice du maximum de la matrice fournie.
* @param matrix La matrice dans laquelle on recherche le mininum.
* @retrun       L'indice unidimensionnel du maximum dans la matrice.
*/
__host__ int argmax(FloatMatrix* matrix);

///////////////////////////// Display functions /////////////////////////////

/**
* Affiche les coefficients d'une matrice. Attention, il faut copier la 
* matrice coté CPU avant de l'afficher.
* @param matrix La matrice en question.
*/
__host__ void printMatrix(FloatMatrix* matrix);

/**
* Affiche une matrice en tant qu'image et ce en ASCII. Attention, il faut
* copier la matrice coté CPU avant de l'afficher. Les coefficents doivent
* se trouver entre 0 et 1.
* @param matrix La matrice en question.
*/
__host__ void displayMatrix(FloatMatrix* matrix);

/**
* Affiche une matrice en tant qu'image et ce en ASCII. Attention, il faut
* copier la matrice coté CPU avant de l'afficher. Les coefficents doivent
* se trouver entre -1 et 1. Le bleu représente des valeurs positives. Le 
* rouge, des valeurs négatives.
* @param matrix La matrice en question.
*/
__host__ void displaySignedMatrix(FloatMatrix* matrix);

/**
* Affiche un vecteur de valeurs entre 0 et 1 comme un graphique à barres.
* @param matrix Le vecteur en question.
* @param height Hauteur du graphique.
* @param title  Titre du graphique.
*/
__host__ void displayVectorAsBarGraph(FloatMatrix* matrix, int height, const char* title);

#endif