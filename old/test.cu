#include <stdio.h>
#include <stdlib.h>



float readFloat(FILE *f) {
    float v;
    fread((void*)(&v), sizeof(v), 1, f);
    return v;
}

int main() {
    FILE* ptr = fopen("test.bin","rb");
    float val;
    
    for (int i=0; i<4; i++) {
    val = readfloat(ptr);
    printf("%f\n", val);
    }

    return 0;
}