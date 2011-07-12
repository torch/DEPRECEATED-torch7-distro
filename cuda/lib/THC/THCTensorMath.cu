#include "THCTensorMath.h"
#include "THCGeneral.h"

#define NB_THREADS_PER_BLOCK 256

__global__ void THCudaTensor_kernel_fill(float *data, float value, long size)
{
  long k = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(k < size)
    data[k] = value;
}

void THCudaTensor_fill(THCudaTensor *self_, float value)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  long nbBlocksPerGrid = (size + NB_THREADS_PER_BLOCK - 1) / NB_THREADS_PER_BLOCK;

  THCudaTensor_kernel_fill<<<nbBlocksPerGrid, NB_THREADS_PER_BLOCK>>>(THCudaTensor_data(self), value, size);

  THCudaTensor_freeCopyTo(self, self_);
}

void THCudaTensor_zero(THCudaTensor *self_)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  cudaMemset(THCudaTensor_data(self), 0, sizeof(float)*THCudaTensor_nElement(self));
  THCudaTensor_freeCopyTo(self, self_);
}

__global__ void THCudaTensor_kernel_add(float *data, float value, long size)
{
  long k = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(k < size)
    data[k] += value;
}

void THCudaTensor_add(THCudaTensor *self_, float value)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  long nbBlocksPerGrid = (size + NB_THREADS_PER_BLOCK - 1) / NB_THREADS_PER_BLOCK;

  THCudaTensor_kernel_add<<<nbBlocksPerGrid, NB_THREADS_PER_BLOCK>>>(THCudaTensor_data(self), value, size);
  
  THCudaTensor_freeCopyTo(self, self_);
}

void THCudaTensor_mul(THCudaTensor *self_, float value)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);

  cublasSscal(THCudaTensor_nElement(self), value, THCudaTensor_data(self), 1);

  THCudaTensor_freeCopyTo(self, self_);
}

void THCudaTensor_div(THCudaTensor *self_, float value)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);

  cublasSscal(THCudaTensor_nElement(self), 1/value, THCudaTensor_data(self), 1);

  THCudaTensor_freeCopyTo(self, self_);
}

void THCudaTensor_cadd(THCudaTensor *self_, float value, THCudaTensor *src)
{
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src), 3, "size do not match");

  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    src = THCudaTensor_newContiguous(src);

    cublasSaxpy(THCudaTensor_nElement(self), value, THCudaTensor_data(src), 1, THCudaTensor_data(self), 1);
                
    THCudaTensor_free(src);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

__global__ void THCudaTensor_kernel_cmul(float *data, float *src, long size)
{
  long k = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(k < size)
    data[k] *= src[k];
}

void THCudaTensor_cmul(THCudaTensor *self_, THCudaTensor *src)
{
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src), 2, "size do not match");

  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    long nbBlocksPerGrid = (size + NB_THREADS_PER_BLOCK - 1) / NB_THREADS_PER_BLOCK;
    src = THCudaTensor_newContiguous(src);
    

    THCudaTensor_kernel_cmul<<<nbBlocksPerGrid, NB_THREADS_PER_BLOCK>>>(THCudaTensor_data(self), THCudaTensor_data(src), size);
                
    THCudaTensor_free(src);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

__global__ void THCudaTensor_kernel_cdiv(float *data, float *src, long size)
{
  long k = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(k < size)
    data[k] /= src[k];
}

void THCudaTensor_cdiv(THCudaTensor *self_, THCudaTensor *src)
{
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src), 1, "size do not match");

  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    long nbBlocksPerGrid = (size + NB_THREADS_PER_BLOCK - 1) / NB_THREADS_PER_BLOCK;
    src = THCudaTensor_newContiguous(src);
    

    THCudaTensor_kernel_cdiv<<<nbBlocksPerGrid, NB_THREADS_PER_BLOCK>>>(THCudaTensor_data(self), THCudaTensor_data(src), size);
                
    THCudaTensor_free(src);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

__global__ void THCudaTensor_kernel_addcmul(float *data, float value, float *src1, float *src2, long size)
{
  long k = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(k < size)
    data[k] += value*src1[k]*src2[k];
}


void THCudaTensor_addcmul(THCudaTensor *self_, float value, THCudaTensor *src1, THCudaTensor *src2)
{
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src1), 3, "size do not match");
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src2), 4, "size do not match");

  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    long nbBlocksPerGrid = (size + NB_THREADS_PER_BLOCK - 1) / NB_THREADS_PER_BLOCK;
    src1 = THCudaTensor_newContiguous(src1);
    src2 = THCudaTensor_newContiguous(src2);
    

    THCudaTensor_kernel_addcmul<<<nbBlocksPerGrid, NB_THREADS_PER_BLOCK>>>(THCudaTensor_data(self), value, THCudaTensor_data(src1), THCudaTensor_data(src2), size);
                
    THCudaTensor_free(src1);
    THCudaTensor_free(src2);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

__global__ void THCudaTensor_kernel_addcdiv(float *data, float value, float *src1, float *src2, long size)
{
  long k = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(k < size)
    data[k] += value*src1[k]/src2[k];
}


void THCudaTensor_addcdiv(THCudaTensor *self_, float value, THCudaTensor *src1, THCudaTensor *src2)
{
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src1), 3, "size do not match");
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src2), 4, "size do not match");

  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    long nbBlocksPerGrid = (size + NB_THREADS_PER_BLOCK - 1) / NB_THREADS_PER_BLOCK;
    src1 = THCudaTensor_newContiguous(src1);
    src2 = THCudaTensor_newContiguous(src2);
    

    THCudaTensor_kernel_addcdiv<<<nbBlocksPerGrid, NB_THREADS_PER_BLOCK>>>(THCudaTensor_data(self), value, THCudaTensor_data(src1), THCudaTensor_data(src2), size);
                
    THCudaTensor_free(src1);
    THCudaTensor_free(src2);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

float THCudaTensor_dot(THCudaTensor *self, THCudaTensor *src)
{
  THArgCheck(THCudaTensor_nElement(self) == THCudaTensor_nElement(src), 2, "size do not match");

  {
    self = THCudaTensor_newContiguous(self);
    src = THCudaTensor_newContiguous(src);

    float result = cublasSdot(THCudaTensor_nElement(self),
                              THCudaTensor_data(self), 1,
                              THCudaTensor_data(src), 1);

    THCudaTensor_free(src);
    THCudaTensor_free(self);

    return result;
  }
}

float THCudaTensor_min(THCudaTensor *self)
{
  int index;
  int result;

  self = THCudaTensor_newContiguous(self);  
  index = cublasIsamin(THCudaTensor_nElement(self), THCudaTensor_data(self), 1);
  result = *(THCudaTensor_data(self)+index);
  THCudaTensor_free(self);

  return result;
}

float THCudaTensor_max(THCudaTensor *self)
{
  int index;
  int result;

  self = THCudaTensor_newContiguous(self);  
  index = cublasIsamax(THCudaTensor_nElement(self), THCudaTensor_data(self), 1);
  result = *(THCudaTensor_data(self)+index);
  THCudaTensor_free(self);

  return result;
}

float THCudaTensor_sum(THCudaTensor *self)
{
  int result;

  self = THCudaTensor_newContiguous(self);  
  result = cublasSasum(THCudaTensor_nElement(self), THCudaTensor_data(self), 1);
  THCudaTensor_free(self);

  return result;
}

