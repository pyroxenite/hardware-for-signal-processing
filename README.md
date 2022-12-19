# Hardware for signal processing
Autheurs : Pharoah Jardin et Pablo Dumenil

## Utilisation
Pour compiler, on utilise la comande suivante dans le répertoire du projet :
```bash
nvcc *.cu
```

Ce qui génère un exécutable `a.out`, que l'on exécute par `./a.out`.

Les sous-commandes suivantes sont disponibles :
 * `classify <path> [-verbose]` qui classifie une image (en binaire `float32`) à l'emplacement `path`. L'option `-verbose` affiche les sorties des couches intermédiares ;
 * `test` qui exécute les testes unitaires ;
 * `demo <demo-name>` qui exécute une démo parmi :
    * `blur` : flou par convolution ;
    * `sobel` : application d'un filtre de Sobel ;
    * `kernal-read` : lecture et affichage de noyaux de convolution ;
    * `image-read` : lecture et affichage de images de teste ;
    * `classify-all` : affiche et classifie une image par chiffre de 0 à 9.

Par exemple :
```sh
./a.out demo classify-all
```

## Organisation du projet

Pour représenter les données dans les couches, on utilise une matrice de flottants. Comme les opérations couteuses ont lieu sur une carte graphique, la gestion de deux tableaux est nécessaire : un accessble depuis le CPU et l'autre depuis le GPU. Pour faciliter cette gestion, la structure `FloatMatrix` est utlisée. Elle stocke, en plus des pointeurs vers ces deux tableau, les dimensions de la matrice. (Voir `matrix.h`.)

Les couches sont représentées par des extensions de la structure `Layer`. Ces structures stockent les données qui représentent la couche (poids, dimensions), un pointeur vers la couche suivante ainsi que les tableaux de sortie de la couche. Un réseau de neuronnes est alors représenté comme une liste chainée de couches. (Voir `layer.h` et `neural-network.h`.)

Des testes unitaires sont écrits dans `unit-tests.cu` et sont exécutable par `./a.out test`.