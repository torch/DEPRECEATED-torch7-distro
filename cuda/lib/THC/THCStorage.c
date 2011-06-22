#include "THCStorage.h"
#include "cublas.h"

TH_API THCudaStorage* THCudaStorage_new(void)
{
  return THStorage_(newWithSize)(0);  
}

TH_API THCudaStorage* THCudaStorage_newWithSize(long size)
{
  THStorage *storage = THAlloc(sizeof(THCudaStorage));
  cudaMalloc((void**)&(storage->data), size * sizeof(float));
  storage->size = size;
  storage->refcount = 1;
  storage->isMapped = 0;
  return storage;
}

TH_API THCudaStorage* THCudaStorage_newWithSize1(real data0)
{
  return NULL;
}

TH_API THCudaStorage* THCudaStorage_newWithSize2(real data0, real data1)
{
  return NULL;
}

TH_API THCudaStorage* THCudaStorage_newWithSize3(real data0, real data1, real data2)
{
  return NULL;
}

TH_API THCudaStorage* THCudaStorage_newWithSize4(real data0, real data1, real data2, real data3)
{
  return NULL;
}

TH_API THCudaStorage* THCudaStorage_newWithMapping(const char *fileName, int isShared)
{
  return NULL;
}
