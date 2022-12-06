#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix-new.h"
#include <math.h>

// int main() {
//     srand(time(NULL));

//     FloatMatrix* raw_data = randomMatrix(32, 32);

//     FloatMatrix** kernals = randomMatrices(6, 5, 5);
//     FloatMatrix** postConvolution = zeroMatrices(6, 28, 28);
//     FloatMatrix** postSubsampling = zeroMatrices(6, 14, 14);

//     convolve(raw_data, kernals[0], postConvolution[0]);

//     freeMatrix(raw_data);
//     forEach(kernals, 6, freeMatrix);
//     forEach(postConvolution, 6, freeMatrix);
//     forEach(postSubsampling, 6, freeMatrix);

//     cudaDeviceSynchronize();
//     return 0;
// }






// int main() {
//     srand(time(NULL));

//     int im_size = 32;
//     int ker_size = 6;
//     int res_size = im_size - ker_size + 1;

//     FloatMatrix* image = zeroMatrix(im_size, im_size);
//     FloatMatrix* kernal = zeroMatrix(ker_size, ker_size);
//     FloatMatrix* result = zeroMatrix(res_size, res_size);

//     drawCircle(image, im_size/2.5, im_size/2.5, im_size/5.0, 0.4);
//     drawCircle(image, 1.5*im_size/2.5, 1.5*im_size/2.5, im_size/5.0, 0.1);
//     drawCircle(kernal, ker_size/2 - 0.5, ker_size/2 - 0.5, ker_size*2.0, 0.1);

//     displayMatrixAsAscii(image);
//     displayMatrixAsAscii(kernal);

//     convolve(image, kernal, result);

//     cudaDeviceSynchronize();

//     displayMatrixAsAscii(result);
//     return 0;
// }


int main() {
    srand(time(NULL));

    int im_size = 32;
    int ker_size = 3;
    int res_size = im_size - ker_size + 1;

    FloatMatrix* image = zeroMatrix(im_size, im_size);
    FloatMatrix* kernal = zeroMatrix(ker_size, ker_size);
    FloatMatrix* result = zeroMatrix(res_size, res_size);

    drawCircle(image, im_size/2.5, im_size/2.5, im_size/5.0, 0.4);
    drawCircle(image, 1.5*im_size/2.5, 1.5*im_size/2.5, im_size/5.0, 0.1);

    for (int i=0; i<ker_size; i++) {
        for (int j=0; j<ker_size; j++) {
            kernal->cpu[i*ker_size + j] = (j - ker_size/2.0 + 0.5)/ker_size*2;
        }
    }
    displayMatrix(image);
    displaySignedMatrix(kernal);

    copyToDevice(image);
    copyToDevice(kernal);

    convolve(image, kernal, result);

    copyFromDevice(result);

    cudaDeviceSynchronize();
    displaySignedMatrix(result);
    
    return 0;
}