#include "THCTensor.h"
#include "THCGeneral.h"
#include "THGeneral.h"

#define NB_THREADS_PER_BLOCK 256

static void THCudaTensor_computesz(THCudaTensor *self, long **sz_, long **st_)
{
  long *sz, *st, *szh;
  int i;
  
  THCudaCheck(cudaMalloc(&sz, sizeof(long)*self->nDimension));
  THCudaCheck(cudaMalloc(&st, sizeof(long)*self->nDimension));
  szh = (long*)THAlloc(sizeof(long)*self->nDimension);

  for(i = self->nDimension-1; i >= 0; i--)
  {
    if(i == self->nDimension-1)
      szh[i] = 1;
    else
      szh[i] = szh[i+1]*self->size[i+1];
  }

  THCudaCheck(cudaMemcpy(sz, szh, self->nDimension * sizeof(long), cudaMemcpyHostToDevice));
  THCudaCheck(cudaMemcpy(st, self->stride, self->nDimension * sizeof(long), cudaMemcpyHostToDevice));
  THFree(szh);

  *sz_ = sz;
  *st_ = st;
}

__global__ void THCudaTensor_kernel_fill(float *data, float value, long size)
{
  long k = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(k < size)
    data[k] = value;
}

__global__ void THCudaTensor_kernel_copy(float *dst, 
                                         long *dst_sz, long *dst_st, int dst_dim,
                                         float *src,
                                         long *src_sz, long *src_st, int src_dim,
                                         long n_elem)
{
  long k = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(k < n_elem)
  {
    long dst_idx = 0;
    long dst_rest = k;
    for(int dim = 0; dim < dst_dim; dim++)
    {
      dst_idx += (dst_rest/dst_sz[dim])*dst_st[dim];
      dst_rest = dst_rest % dst_sz[dim];
    }

    long src_idx = 0;
    long src_rest = k;
    for(int dim = 0; dim < src_dim; dim++)
    {
      src_idx += (src_rest/src_sz[dim])*src_st[dim];
      src_rest = src_rest % src_sz[dim];
    }

    dst[dst_idx] = src[src_idx];
  }
}

void THCudaTensor_fill(THCudaTensor *self, float value)
{
  long size = THCudaTensor_nElement(self);

  long nbBlocksPerGrid = (size + NB_THREADS_PER_BLOCK - 1) / NB_THREADS_PER_BLOCK;
  THCudaTensor_kernel_fill<<<nbBlocksPerGrid, NB_THREADS_PER_BLOCK>>>(THCudaTensor_data(self), value, size);
}

void THCudaTensor_copy(THCudaTensor *self, THCudaTensor *src)
{
  THArgCheck(THCudaTensor_nElement(self) == THCudaTensor_nElement(src), 2, "sizes do not match"); 

  if(THCudaTensor_isContiguous(self) && THCudaTensor_isContiguous(src))
    THCudaCheck(cudaMemcpy(self->storage->data + self->storageOffset, src->storage->data + src->storageOffset, THCudaTensor_nElement(src) * sizeof(float), cudaMemcpyDeviceToDevice));
  else
  {    
    long *d_self_sz, *d_self_st, *d_src_sz, *d_src_st;
    long nElement = THCudaTensor_nElement(self);
    long nbBlocksPerGrid = (nElement + NB_THREADS_PER_BLOCK - 1) / NB_THREADS_PER_BLOCK;

    THCudaTensor_computesz(self, &d_self_sz, &d_self_st);
    THCudaTensor_computesz(src, &d_src_sz, &d_src_st);
    
    THCudaTensor_kernel_copy<<<nbBlocksPerGrid, NB_THREADS_PER_BLOCK>>>(THCudaTensor_data(self), 
                                                                        d_self_sz, d_self_st, self->nDimension,
                                                                        THCudaTensor_data(src),
                                                                        d_src_sz, d_src_st, src->nDimension,
                                                                        nElement);

    THCudaCheck(cudaFree(d_self_sz));
    THCudaCheck(cudaFree(d_self_st));
    THCudaCheck(cudaFree(d_src_sz));
    THCudaCheck(cudaFree(d_src_st));
  }
}
