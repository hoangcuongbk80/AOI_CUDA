#include "cuda_test.h"

 unsigned char *X, *Y, *Z, *W;
 unsigned char *d_X, *d_Y, *d_Z, *d_W;
 unsigned char *d_offset;

void CPUMemAlloc()
{
  X = (unsigned char*)malloc(MAT_NUMEL * sizeof(unsigned char));
  Y = (unsigned char*)malloc(TOTAL_NUMEL * sizeof(unsigned char));
  Z = (unsigned char*)malloc(TOTAL_NUMEL * sizeof(unsigned char));
  W = (unsigned char*)malloc(TOTAL_NUMEL * sizeof(unsigned char));
}

void GPUMemAlloc()
{
  size_t size = MAT_NUMEL * sizeof(unsigned char);
  cudaError_t msgErr = cudaMalloc(&d_X, size);
  printf("Memory Allocate d_X: %s\n",cudaGetErrorString(msgErr));

  size = TOTAL_NUMEL * sizeof(unsigned char);
  msgErr = cudaMalloc(&d_Y, size);
  printf("Memory Allocate d_Y: %s\n",cudaGetErrorString(msgErr));

  size = TOTAL_NUMEL * sizeof(unsigned char);
  msgErr = cudaMalloc(&d_Z, size);
  printf("Memory Allocate d_Z: %s\n",cudaGetErrorString(msgErr));

  size = TOTAL_NUMEL * sizeof(unsigned char);
  msgErr = cudaMalloc(&d_W, size);
  printf("Memory Allocate d_W: %s\n",cudaGetErrorString(msgErr));
}

void dataGenerating()
{
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
  printf("\n\n");
}

void copyLookupTabGPU()
{
  size_t size = MAT_NUMEL * sizeof(unsigned char);
  cudaError_t msgErr = cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
  printf("cudaMemcpy d_X: %s\n",cudaGetErrorString(msgErr));
  size = TOTAL_NUMEL * sizeof(unsigned char);
  msgErr = cudaMemcpy(d_Y, Y, size, cudaMemcpyHostToDevice);
  printf("cudaMemcpy d_Y: %s\n",cudaGetErrorString(msgErr));

  unsigned char *offset;
  offset = (unsigned char*)malloc(NUM_MATR*sizeof(unsigned char));
  cudaMalloc(&d_offset, NUM_MATR*sizeof(unsigned char)); 
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
   msgErr = cudaMemcpy(d_offset, offset, NUM_MATR*sizeof(unsigned char), cudaMemcpyHostToDevice);
}

void copyDataToGPU()
{
  size_t size = TOTAL_NUMEL * sizeof(unsigned char);
  cudaError_t msgErr = cudaMemcpy(d_Z, Z, size, cudaMemcpyHostToDevice);
}

void copyDataFromGPU()
{
  size_t size = TOTAL_NUMEL * sizeof(unsigned char);
  cudaMemcpy(W, d_W, size, cudaMemcpyDeviceToHost);
}

__global__ void GPU_First_Task(unsigned char *offset, unsigned char *X, unsigned char *Z, unsigned char *W) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int depth = index / (MAT_NUMEL);
  int X_index = index % MAT_NUMEL;
  W[index] = Z[index] + X[X_index] * offset[depth]; 
}

__global__ void GPU_Second_Task(unsigned char *offset, unsigned char *Y, unsigned char *Z, unsigned char *W) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int depth = index / (MAT_NUMEL);
  W[index] = Z[index] + offset[depth];
}

int main()
{
  clock_t start, end;
  double cpu_time_used;
  CPUMemAlloc();
  dataGenerating();
  GPUMemAlloc();
  copyLookupTabGPU();

//----Speed of Vector Product/Multiplication-------//
  printf("\nSpeed of Vector Product/Multiplication:\n");

  start = clock();
  for (int i = 0; i < NUM_LOOP; i++) copyDataToGPU();
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("copyDataToGPU took %f seconds to execute \n", cpu_time_used);

  start = clock();
  for(int i = 0; i < NUM_LOOP; i++)
  {
     GPU_First_Task<<<MAT_SIZE*MAT_SIZE, NUM_MATR>>>(d_offset, d_X, d_Z, d_W);
     cudaThreadSynchronize();
  }
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Kernel took %f seconds to execute \n", cpu_time_used);

  start = clock();
  for (int i = 0; i < NUM_LOOP; i++) copyDataFromGPU();
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("copyDataFromGPU took %f seconds to execute \n", cpu_time_used);

  printf("Matrix W: ");
  for(int i = 0; i < 5; i++) printf("%d ", W[i]);
  printf("... ");
  for(int i = 0; i < 5; i++) printf("%d ", W[MAT_NUMEL * NUM_MATR - i -1]);

//-----------------Speed of Addition----------------//
  printf("\n\nSpeed of Addition using look-up table:\n");
  start = clock();
  for (int i = 0; i < NUM_LOOP; i++) copyDataToGPU();
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("copyDataToGPU took %f seconds to execute \n", cpu_time_used);

  start = clock();
  for(int i = 0; i < NUM_LOOP; i++)
  {
     GPU_Second_Task<<<MAT_SIZE*MAT_SIZE, NUM_MATR>>>(d_offset, d_Y, d_Z, d_W);
     cudaThreadSynchronize();
  }
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Kernel took %f seconds to execute \n", cpu_time_used);

  start = clock();
  for (int i = 0; i < NUM_LOOP; i++) copyDataFromGPU();
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("copyDataFromGPU took %f seconds to execute \n", cpu_time_used);

  printf("Matrix W: ");
  for(int i = 0; i < 5; i++) printf("%d ", W[i]);
  printf("... ");
  for(int i = 0; i < 5; i++) printf("%d ", W[MAT_NUMEL * NUM_MATR - i -1]);
  printf("\n");

// Free device memory
  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_Z);
  cudaFree(d_W);
  cudaFree(d_offset);
// Free host memory
  free(X);
  free(Y);
  free(Z);
  free(W);

  return 0;
}


