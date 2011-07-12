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

void THCudaTensor_addmv(THCudaTensor *self_, float alpha, THCudaTensor *mat, THCudaTensor *vec)
{
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected");
 
  if( mat->size[1] != vec->size[0] )
    THError("size mismatch");
  
  if(self_->nDimension != 1)
    THError("size mismatch");
    
  if( self_->size[0] != mat->size[0] )
    THError("size mismatch");

  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    mat = THCudaTensor_newContiguous(mat);
    vec = THCudaTensor_newContiguous(vec);

    cublasSgemv('t',  mat->size[1], mat->size[0],
                alpha, THCudaTensor_data(mat), mat->stride[0],
                THCudaTensor_data(vec), vec->stride[0],
                1, THCudaTensor_data(self), self->stride[0]);

    THCudaTensor_free(mat);
    THCudaTensor_free(vec);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

void THCudaTensor_addmm(THCudaTensor *self_, float alpha, THCudaTensor *mat1, THCudaTensor *mat2)
{
  if( (mat1->nDimension != 2) || (mat2->nDimension != 2) ) 
    THError("matrix and matrix expected"); 
 
  if(self_->nDimension != 2)
    THError("size mismatch"); 

  if( (self_->size[0] != mat1->size[0]) || (self_->size[1] != mat2->size[1]) || (mat1->size[1] != mat2->size[0]) ) 
    THError("size mismatch"); 

  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    mat1 = THCudaTensor_newContiguous(mat1);
    mat2 = THCudaTensor_newContiguous(mat2);

    cublasSgemm('n',
                'n',
                self->size[1],
                self->size[0],
                mat2->size[0],
                alpha,
                THCudaTensor_data(mat2),
                mat2->stride[0],
                THCudaTensor_data(mat1),
                mat1->stride[0],
                1,
                THCudaTensor_data(self),
                self->stride[0]);
    
    THCudaTensor_free(mat1);
    THCudaTensor_free(mat2);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

void THCudaTensor_addr(THCudaTensor *self_, float alpha, THCudaTensor *vec1, THCudaTensor *vec2)
{
  if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected");

  if(self_->nDimension != 2)
    THError("size mismatch");
    
  if( (self_->size[0] != vec1->size[0]) || (self_->size[1] != vec2->size[0]) )
    THError("size mismatch");

  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    vec1 = THCudaTensor_newContiguous(vec1);
    vec2 = THCudaTensor_newContiguous(vec2);

    cublasSger(vec2->size[0], vec1->size[0],
               alpha, THCudaTensor_data(vec2), vec2->stride[0],
               THCudaTensor_data(vec1), vec1->stride[0],
               THCudaTensor_data(self), self->stride[0]);
    
    THCudaTensor_free(vec1);
    THCudaTensor_free(vec2);
    THCudaTensor_freeCopyTo(self, self_);
  }  
}

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(NAME, CFUNC)                   \
  __global__ void THCudaTensor_kernel_##NAME(float *data, long size)       \
  {                                                                     \
    long k = blockDim.x * blockIdx.x + threadIdx.x;                     \
                                                                        \
    if(k < size)                                                        \
      data[k] = CFUNC(data[k]);                                           \
  }                                                                     \
                                                                        \
  void THCudaTensor_##NAME(THCudaTensor *self_)                         \
  {                                                                     \
    THCudaTensor *self = THCudaTensor_newContiguous(self_);             \
    long size = THCudaTensor_nElement(self);                            \
    long nbBlocksPerGrid = (size + NB_THREADS_PER_BLOCK - 1) / NB_THREADS_PER_BLOCK; \
                                                                        \
    THCudaTensor_kernel_##NAME<<<nbBlocksPerGrid, NB_THREADS_PER_BLOCK>>>(THCudaTensor_data(self), size); \
                                                                        \
    THCudaTensor_freeCopyTo(self, self_);                               \
  }

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log, log)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log1p, log1p)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(exp, exp)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cos, cos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(acos, acos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cosh, cosh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sin, sin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(asin, asin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sinh, sinh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tan, tan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(atan, atan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tanh, tanh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sqrt, sqrt)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(ceil, ceil)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(floor, floor)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(abs, fabs)

__global__ void THCudaTensor_kernel_pow(float *data, float value, long size)
{
  long k = blockDim.x * blockIdx.x + threadIdx.x;

  if(k < size)
    data[k] = pow(data[k], value);
}

void THCudaTensor_pow(THCudaTensor *self_, float value)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  long nbBlocksPerGrid = (size + NB_THREADS_PER_BLOCK - 1) / NB_THREADS_PER_BLOCK;

  THCudaTensor_kernel_pow<<<nbBlocksPerGrid, NB_THREADS_PER_BLOCK>>>(THCudaTensor_data(self), value, size);

  THCudaTensor_freeCopyTo(self, self_);
}

float THCudaTensor_mean(THCudaTensor *self)
{
  return -1;
}

float THCudaTensor_var(THCudaTensor *self)
{
  return -1;
}

float THCudaTensor_std(THCudaTensor *self)
{
  return -1;
}

float THCudaTensor_norm(THCudaTensor *self, float value)
{
  return -1;
}

float THCudaTensor_dist(THCudaTensor *self, THCudaTensor *src, float value)
{
  return -1;
}
