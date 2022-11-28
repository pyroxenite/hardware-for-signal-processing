#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include <math.h>

int main()
{
    srand(time(NULL));

    float* raw_data = randomMatrix(32, 32);

    int l = 32*32;

    for (int i=0; i<l; i++) {
        int x = i % 32;
        int y = i / 32;
        raw_data[i] = (1+sin(sqrt(x*x + y*y)/2.0))/2;
    }

    displayMatrixAsAscii(raw_data, 32, 32);

    return 0;
}