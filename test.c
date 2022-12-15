#include <stdio.h>
#include <stdlib.h>

typedef struct A {
    int x;
    int y;
} A;

typedef struct B {
    int x;
    int y;
    int z;
} B;

void printA(A* a) {
    printf("A(x=%d, y=%d)\n", a->x, a->y);
}

void printB(B* b) {
    printf("B(x=%d, y=%d, z=%d)\n", b->x, b->y, b->z);
}

int main() {
    B* b = (B*) malloc(sizeof(B));
    b->x = 3;
    b->y = 4;
    b->z = 5;

    A* a = (A*) b;

    B* backToB = (B*) a;

    printB(b);
    printA(a);
    printB(backToB);

    printf("b->z = %x\n", &(b->z));
    printf("backToB->z = %x\n", &(backToB->z));

    return 0;
}