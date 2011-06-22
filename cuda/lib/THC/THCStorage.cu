#include "THCStorage.h"

#define NB_THREADS_PER_BLOCK 256

__global__ void Storage_kernel_fill(float *data, float value, long size)
{
  long i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < size)
  	data[i] = value;
}

void THCudaStorage_fill(THCudaStorage *storage, real value)
{
  long nbBlocksPerGrid = (storage->size + NB_THREADS_PER_BLOCK - 1) / NB_THREADS_PER_BLOCK;
  Storage_kernel_fill<<<nbBlocksPerGrid, NB_THREADS_PER_BLOCK>>>(storage->data, value, storage->size);
}
