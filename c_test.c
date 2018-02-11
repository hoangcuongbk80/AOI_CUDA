#include "c_test.h"

void computeOffset(unsigned char *Z, unsigned char *offset)
{
  for (int depth = 0; depth < NUM_MATR; depth++)
  {
    unsigned char maxval = 0;
    for(int row = 0; row < MAT_SIZE; row++)
     for(int col = 0; col < MAT_SIZE; col++)
     {
       int index = depth * MAT_NUMEL + row * MAT_SIZE + col;
       if(Z[index] > maxval) maxval = Z[index];
     }
    offset[depth] = 255- maxval;
   }
}

void First_Task(unsigned char *X, unsigned char *Z, unsigned char *W, unsigned char *offset) 
{
  for(int depth = 0; depth < NUM_MATR; depth++)
  {
    for(int row = 0; row < MAT_SIZE; row++)
     for(int col = 0; col < MAT_SIZE; col++)
     {
       int index = depth * MAT_NUMEL + row * MAT_SIZE + col;
       W[index] = Z[index] + X[row * MAT_SIZE + col] * offset[depth]; 
     }
  }
}

void Second_Task(unsigned char *Y, unsigned char *Z, unsigned char *W, unsigned char *offset) 
{
  for(int depth = 0; depth < NUM_MATR; depth++)
  {
    for(int row = 0; row < MAT_SIZE; row++)
     for(int col = 0; col < MAT_SIZE; col++)
     {
       int index = depth * MAT_NUMEL + row * MAT_SIZE + col;
       W[index] = Z[index] + offset[depth];
     }
  }
}

int main()
{
  clock_t start, end;
  double cpu_time_used;
  unsigned char *X, *Y, *Z, *W, *offset;

  X = (unsigned char*)malloc(MAT_NUMEL * sizeof(unsigned char));
  Y = (unsigned char*)malloc(MAT_NUMEL * NUM_MATR * sizeof(unsigned char));
  Z = (unsigned char*)malloc(MAT_NUMEL * NUM_MATR * sizeof(unsigned char));
  W = (unsigned char*)malloc(MAT_NUMEL * NUM_MATR * sizeof(unsigned char));
  offset = (unsigned char*)malloc(NUM_MATR*sizeof(unsigned char)); 
  
// (1) look-up table
  for(int row = 0; row < MAT_SIZE; row++)
   for(int col = 0; col < MAT_SIZE; col++)
    X[row * MAT_SIZE + col] = 1;

  for(int depth = 0; depth < NUM_MATR; depth++)
   for(int row = 0; row < MAT_SIZE; row++)
    for(int col = 0; col < MAT_SIZE; col++)
     Y[depth * MAT_NUMEL + row * MAT_SIZE + col] = depth;

//(2) assign random values to 256 matrices 100x100
  for(int depth = 0; depth < NUM_MATR; depth++)  
   for(int row = 0; row < MAT_SIZE; row++)
    for(int col = 0; col < MAT_SIZE; col++)
    {
      int index = depth * MAT_NUMEL + row * MAT_SIZE + col;
      Z[index] = rand() % 233;//256
      W[index] = Z[index];
    }
  printf("Matrix W: ");
  for(int i = 0; i < 5; i++) printf("%d ", W[i]);
  printf("... ");
  for(int i = 0; i < 5; i++) printf("%d ", W[MAT_NUMEL * NUM_MATR - i -1]);

  computeOffset(Z, offset);

// (3) Fist task: Speed of Vector Product/Multiplication
  start = clock();
  for(int i = 0; i < NUM_LOOP; i++)
   First_Task(X, Z, W, offset);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("\n1st task took %f seconds to execute \n", cpu_time_used);
  printf("Matrix W: ");
  for(int i = 0; i < 5; i++) printf("%d ", W[i]);
  printf("... ");
  for(int i = 0; i < 5; i++) printf("%d ", W[MAT_NUMEL * NUM_MATR - i -1]);

// (4) Second task: Speed of Addition
  start = clock();
  for(int i = 0; i < NUM_LOOP; i++)
   Second_Task(Y, Z, W, offset);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("\n2nd task took %f seconds to execute \n", cpu_time_used);
  printf("Matrix W: ");
  for(int i = 0; i < 5; i++) printf("%d ", W[i]);
  printf("... ");
  for(int i = 0; i < 5; i++) printf("%d ", W[MAT_NUMEL * NUM_MATR - i -1]);

  printf("\n");

  return 0;
}
