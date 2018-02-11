#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAT_SIZE 100
#define MAT_NUMEL 10000
#define TOTAL_NUMEL 2560000
#define NUM_MATR 256
#define NUM_LOOP 200

__global__ void GPU_First_Task(unsigned char *offset, unsigned char *X, unsigned char *Z, unsigned char *W);
__global__ void GPU_Second_Task(unsigned char *offset, unsigned char *Y, unsigned char *Z, unsigned char *W);
