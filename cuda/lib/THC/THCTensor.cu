#include "THCTensor.h"

#define NB_THREADS_PER_BLOCK 256

__global__ void THCudaTensor_kernel_fillX(float *data, float value, long size)
{
  long k = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(k < size)
    data[k] = value;
}

__global__ void THCudaTensor_kernel_fill(float *data, float value, long size,
                                         long sz0, long sz1, long sz2, long sz3,
                                         long st0, long st1, long st2, long st3)
{
  long k = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(k < size)
  { 
    long idx = 0;
    long rest = k;

    if(sz0)
    {
      idx += (rest/sz0)*st0;
      rest -= rest % sz0;
    }

    if(sz1)
    {
      idx += (rest/sz1)*st1;
      rest -= rest % sz1;      
    }

    if(sz2)
    {
      idx += (rest/sz2)*st2;
      rest -= rest % sz2;      
    }

    if(sz3)
    {
      idx += (rest/sz3)*st3;
      rest -= rest % sz3;      
    }

    data[idx] = value;
  }
}

void THCudaTensor_fillX(THCudaTensor *self, float value)
{
  long size = THCudaTensor_nElement(self);

  long nbBlocksPerGrid = (size + NB_THREADS_PER_BLOCK - 1) / NB_THREADS_PER_BLOCK;
  THCudaTensor_kernel_fillX<<<nbBlocksPerGrid, NB_THREADS_PER_BLOCK>>>(THCudaTensor_data(self), value, size);
}

void THCudaTensor_fill(THCudaTensor *self, float value)
{
  long size = THCudaTensor_nElement(self);
  long sz0 = 0, sz1 = 0, sz2 = 0, sz3 = 0, st0 = 0, st1 = 0, st2 = 0, st3 = 0;

  if(self->nDimension == 1)
  {
    sz0 = 1;

    st0 = self->stride[0];
  }
  else if(self->nDimension == 2)
  {
    sz1 = 1;
    sz0 = self->size[1];

    st1 = self->stride[1];
    st0 = self->stride[0];
  }
  else if(self->nDimension == 3)
  {
    sz2 = 1;
    sz1 = self->size[2];
    sz0 = sz1*self->size[1];

    st2 = self->stride[2];
    st1 = self->stride[1];
    st0 = self->stride[0];
  }
  else if(self->nDimension == 4)
  {
    sz3 = 1;
    sz2 = self->size[3];
    sz1 = sz2*self->size[2];
    sz0 = sz1*self->size[1];

    st3 = self->stride[3];
    st2 = self->stride[2];
    st1 = self->stride[1];
    st0 = self->stride[0];
  }
  long nbBlocksPerGrid = (size + NB_THREADS_PER_BLOCK - 1) / NB_THREADS_PER_BLOCK;
  THCudaTensor_kernel_fill<<<nbBlocksPerGrid, NB_THREADS_PER_BLOCK>>>(THCudaTensor_data(self), value, size,
                                                                      sz0, sz1, sz2, sz3,
                                                                      st0, st1, st2, st3);
}

void THByteTensor_copyCuda(THByteTensor *self, THCudaTensor *src)
{
}

void THCharTensor_copyCuda(THCharTensor *self, THCudaTensor *src)
{
}

void THShortTensor_copyCuda(THShortTensor *self, THCudaTensor *src)
{
}

void THIntTensor_copyCuda(THIntTensor *self, THCudaTensor *src)
{
}

void THLongTensor_copyCuda(THLongTensor *self, THCudaTensor *src)
{
}

void THFloatTensor_copyCuda(THFloatTensor *self, THCudaTensor *src)
{
}

void THDoubleTensor_copyCuda(THDoubleTensor *self, THCudaTensor *src)
{
}

void THCudaTensor_copyCuda(THCudaTensor *self, THCudaTensor *src)
{
}
