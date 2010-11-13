#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensor.c"
#else

static void THTensor_(reinit)(THTensor *tensor, THStorage *storage, long storageOffset, int nDimension, long *size, long *stride);

/* Empty init */
THTensor *THTensor_(new)(void)
{
  THTensor *tensor = THAlloc(sizeof(THTensor));
  tensor->size = NULL;
  tensor->stride = NULL;
  tensor->nDimension = 0;
  tensor->storage = NULL;
  tensor->storageOffset = 0;
  tensor->ownStorage = 0;
  tensor->refcount = 1;
  return tensor;
}

/* Pointer-copy init */
THTensor *THTensor_(newWithTensor)(THTensor *src)
{
  THTensor *tensor = THTensor_(new)();
  THTensor_(reinit)(tensor, src->storage, src->storageOffset, src->nDimension, src->size, src->stride);
  return tensor;
}

/* Storage init */
THTensor *THTensor_(newWithStorage)(THStorage *storage, long storageOffset, int nDimension, long *size, long *stride)
{
  THTensor *tensor = THTensor_(new)();
  THTensor_(reinit)(tensor, storage, storageOffset, nDimension, size, stride);
  return tensor;
}

THTensor *THTensor_(newWithStorage1d)(THStorage *storage, long storageOffset,
                               long size0, long stride0)
{
  return THTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, -1, -1,  -1, -1,  -1, -1);
}

THTensor *THTensor_(newWithStorage2d)(THStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1)
{
  return THTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, size1, stride1,  -1, -1,  -1, -1);
}

THTensor *THTensor_(newWithStorage3d)(THStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2)
{
  return THTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, size1, stride1,  size2, stride2,  -1, -1);
}

THTensor *THTensor_(newWithStorage4d)(THStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2,
                               long size3, long stride3)
{
  long size[4];
  long stride[4];
  size[0] = size0;
  size[1] = size1;
  size[2] = size2;
  size[3] = size3;
  stride[0] = stride0;
  stride[1] = stride1;
  stride[2] = stride2;
  stride[3] = stride3;  
  return THTensor_(newWithStorage)(storage, storageOffset, 4, size, stride);
}

/* Normal init */
THTensor *THTensor_(newWithSize)(int nDimension, long *size, long *stride)
{
  return THTensor_(newWithStorage)(NULL, 0, nDimension, size, stride);
}

THTensor *THTensor_(newWithSize1d)(long size0)
{
  return THTensor_(newWithSize4d)(size0, -1, -1, -1);
}

THTensor *THTensor_(newWithSize2d)(long size0, long size1)
{
  return THTensor_(newWithSize4d)(size0, size1, -1, -1);
}

THTensor *THTensor_(newWithSize3d)(long size0, long size1, long size2)
{
  return THTensor_(newWithSize4d)(size0, size1, size2, -1);
}

THTensor *THTensor_(newWithSize4d)(long size0, long size1, long size2, long size3)
{
  long size[4];
  size[0] = size0;
  size[1] = size1;
  size[2] = size2;
  size[3] = size3;
  return THTensor_(newWithSize)(4, size, NULL);
}

TH_API THTensor *THTensor_(newContiguous)(THTensor *tensor)
{
  return NULL;
}

/* Set */
void THTensor_(setTensor)(THTensor *tensor, THTensor *src)
{
  THTensor_(reinit)(tensor, src->storage, src->storageOffset, src->nDimension, src->size, src->stride);
}

void THTensor_(setStorage)(THTensor *tensor, THStorage *storage, long storageOffset, int nDimension, long *size, long *stride)
{
  THTensor_(reinit)(tensor, storage, storageOffset, nDimension, size, stride);
}

void THTensor_(setStorage1d)(THTensor *tensor, THStorage *storage, long storageOffset,
                        long size0, long stride0)
{
  THTensor_(setStorage4d)(tensor, storage, storageOffset, size0, stride0, -1, -1, -1, -1, -1, -1);
}

void THTensor_(setStorage2d)(THTensor *tensor, THStorage *storage, long storageOffset,
                        long size0, long stride0,
                        long size1, long stride1)
{
  THTensor_(setStorage4d)(tensor, storage, storageOffset, size0, stride0, size1, stride1, -1, -1, -1, -1);
}

void THTensor_(setStorage3d)(THTensor *tensor, THStorage *storage, long storageOffset,
                        long size0, long stride0,
                        long size1, long stride1,
                        long size2, long stride2)
{
  THTensor_(setStorage4d)(tensor, storage, storageOffset, size0, stride0, size1, stride1, size2, stride2, -1, -1);
}

void THTensor_(setStorage4d)(THTensor *tensor, THStorage *storage, long storageOffset,
                        long size0, long stride0,
                        long size1, long stride1,
                        long size2, long stride2,
                        long size3, long stride3)
{
  long size[4];
  long stride[4];
  size[0] = size0;
  size[1] = size1;
  size[2] = size2;
  size[3] = size3;
  stride[0] = stride0;
  stride[1] = stride1;
  stride[2] = stride2;
  stride[3] = stride3;
  THTensor_(setStorage)(tensor, storage, storageOffset, 4, size, stride);
}

void THTensor_(set1d)(THTensor *tensor, long x0, real value)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]] = value;
}

real THTensor_(get1d)(THTensor *tensor, long x0)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  return (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]];
}

void THTensor_(set2d)(THTensor *tensor, long x0, long x1, real value)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]+x1*tensor->stride[1]] = value;
}

real THTensor_(get2d)(THTensor *tensor, long x0, long x1)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  return( (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]+x1*tensor->stride[1]] );
}

void THTensor_(set3d)(THTensor *tensor, long x0, long x1, long x2, real value)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]] = value;
}

real THTensor_(get3d)(THTensor *tensor, long x0, long x1, long x2)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  return( (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]] );
}

void THTensor_(set4d)(THTensor *tensor, long x0, long x1, long x2, long x3, real value)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3]] = value;
}

real THTensor_(get4d)(THTensor *tensor, long x0, long x1, long x2, long x3)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  return( (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3]] );
}

/* Resize */
void THTensor_(resizeAs)(THTensor *tensor, THTensor *src)
{
  int isSame = 0;
  int d;
  if(tensor->nDimension == src->nDimension)
  {
    isSame = 1;
    for(d = 0; d < tensor->nDimension; d++)
    {
      if(tensor->size[d] != src->size[d])
      {
        isSame = 0;
        break;
      }
    }
  }
  if(!isSame)
    THTensor_(reinit)(tensor, NULL, 0, src->nDimension, src->size, NULL);
}

void THTensor_(resize)(THTensor *tensor, int nDimension, long *size)
{
  int isSame = 0;
  if(nDimension == tensor->nDimension)
  {
    int d;
    isSame = 1;
    for(d = 0; d < tensor->nDimension; d++)
    {
      if(tensor->size[d] != size[d])
      {
        isSame = 0;
        break;
      }
    }
  }
  if(!isSame)
    THTensor_(reinit)(tensor, NULL, 0, nDimension, size, NULL);
}

void THTensor_(resize1d)(THTensor *tensor, long size0)
{
  THTensor_(resize4d)(tensor, size0, -1, -1, -1);
}

void THTensor_(resize2d)(THTensor *tensor, long size0, long size1)
{
  THTensor_(resize4d)(tensor, size0, size1, -1, -1);
}

void THTensor_(resize3d)(THTensor *tensor, long size0, long size1, long size2)
{
  THTensor_(resize4d)(tensor, size0, size1, size2, -1);
}

void THTensor_(resize4d)(THTensor *tensor, long size0, long size1, long size2, long size3)
{
  long size[4];
  size[0] = size0;
  size[1] = size1;
  size[2] = size2;
  size[3] = size3;
  THTensor_(resize)(tensor, 4, size);
}

void THTensor_(narrow)(THTensor *tensor, THTensor *src, int dimension, long firstIndex, long size)
{
  if(!src)
    src = tensor;

  THArgCheck( (dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < src->size[dimension]), 3, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= src->size[dimension]), 4, "out of range");

  THTensor_(setTensor)(tensor, src);
  
  if(firstIndex > 0)
    tensor->storageOffset += firstIndex*tensor->stride[dimension];

  tensor->size[dimension] = size;
}


void THTensor_(select)(THTensor *tensor, THTensor *src, int dimension, long sliceIndex)
{
  int d;

  if(!src)
    src = tensor;

  THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 3, "out of range");

  THTensor_(narrow)(tensor, src, dimension, sliceIndex, 1);
  for(d = dimension; d < tensor->nDimension-1; d++)
  {
    tensor->size[d] = src->size[d+1];
    tensor->stride[d] = src->stride[d+1];
  }
  tensor->nDimension--;
}


void THTensor_(transpose)(THTensor *tensor, int dimension1, int dimension2)
{
  long z;

  THArgCheck( (dimension1 >= 0) && (dimension1 < tensor->nDimension), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < tensor->nDimension), 2, "out of range");

  if(dimension1 == dimension2)
	  return;
 
  z = tensor->stride[dimension1];
  tensor->stride[dimension1] = tensor->stride[dimension2];
  tensor->stride[dimension2] = z;
  z = tensor->size[dimension1];
  tensor->size[dimension1] = tensor->size[dimension2];
  tensor->size[dimension2] = z;
}

void THTensor_(unfold)(THTensor *tensor, THTensor *src, int dimension, long size, long step)
{
  long *newSize;
  long *newStride;
  int d;

  if(!src)
    src = tensor;

  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < src->nDimension, 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THTensor_(setTensor)(tensor, src);

  newSize = THAlloc(sizeof(long)*(tensor->nDimension+1));
  newStride = THAlloc(sizeof(long)*(tensor->nDimension+1));

  newSize[tensor->nDimension] = size;
  newStride[tensor->nDimension] = tensor->stride[dimension];
  for(d = 0; d < tensor->nDimension; d++)
  {
    if(d == dimension)
    {
      newSize[d] = (tensor->size[d] - size) / step + 1;
      newStride[d] = step*tensor->stride[d];
    }
    else
    {
      newSize[d] = tensor->size[d];
      newStride[d] = tensor->stride[d];
    }
  }

  THTensor_(reinit)(tensor, tensor->storage, tensor->storageOffset, tensor->nDimension+1, newSize, newStride);
  THFree(newSize);
  THFree(newStride);
}

/* is contiguous? [a bit like in TnXIterator] */
int THTensor_(isContiguous)(THTensor *tensor)
{
  long z = 1;
  int d;
  for(d = 0; d < tensor->nDimension; d++)
  {
    if(tensor->stride[d] == z)
      z *= tensor->size[d];
    else
      return 0;
  }
  return 1;
}

long THTensor_(nElement)(THTensor *tensor)
{
  if(tensor->nDimension == 0)
    return 0;
  else
  {
    long nElement = 1;
    int d;
    for(d = 0; d < tensor->nDimension; d++)
      nElement *= tensor->size[d];
    return nElement;
  }
}

void THTensor_(retain)(THTensor *tensor)
{
  if(tensor)
    ++tensor->refcount;
}

void THTensor_(free)(THTensor *tensor)
{
  if(!tensor)
    return;

  if(--tensor->refcount == 0)
  {
    THFree(tensor->size);
    THFree(tensor->stride);
    THStorage_(free)(tensor->storage);
    THFree(tensor);
  }
}

/*******************************************************************************/

/* This one does everything except coffee */
static void THTensor_(reinit)(THTensor *tensor, THStorage *storage, long storageOffset, int nDimension, long *size, long *stride)
{
  int d;
  int nDimension_;
  long totalSize;

  /* Storage stuff */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");

  tensor->storageOffset = storageOffset;

  if(storage) /* new storage ? */
  {
    if(storage != tensor->storage) /* really new?? */
    {
      if(tensor->storage)
        THStorage_(free)(tensor->storage);
      
      tensor->storage = storage;
      THStorage_(retain)(tensor->storage);
      tensor->ownStorage = 0;
    } /* else we had already this storage, so we keep it */
  }
  else
  {
    if(tensor->storage)
    {
      if(!tensor->ownStorage)
      {
        THStorage_(free)(tensor->storage);
        tensor->storage = THStorage_(new)();
        tensor->ownStorage = 1;
      } /* else we keep our storage */
    }
    else
    {
      tensor->storage = THStorage_(new)();
      tensor->ownStorage = 1;
    }
  }

  /* nDimension, size and stride */
  nDimension_ = 0;
  for(d = 0; d < nDimension; d++)
  {
    if(size[d] > 0)
      nDimension_++;
    else
      break;
  }
  nDimension = nDimension_;

  if(nDimension > 0)
  {
    if(nDimension > tensor->nDimension)
    {
      THFree(tensor->size);
      THFree(tensor->stride);
      tensor->size = THAlloc(sizeof(long)*nDimension);
      tensor->stride = THAlloc(sizeof(long)*nDimension);
    }
    tensor->nDimension = nDimension;

    totalSize = 1;
    for(d = 0; d < tensor->nDimension; d++)
    {
      tensor->size[d] = size[d];
      if(stride && (stride[d] >= 0) )
        tensor->stride[d] = stride[d];
      else
      {
        if(d == 0)
          tensor->stride[d] = 1;
        else
          tensor->stride[d] = tensor->size[d-1]*tensor->stride[d-1];
      }
      totalSize += (tensor->size[d]-1)*tensor->stride[d];
    }
    
    if(totalSize+storageOffset > tensor->storage->size) /* if !ownStorage, that might be a problem! */
    {
      if(!tensor->ownStorage)
        THError("Tensor: trying to resize a storage which is not mine");
      THStorage_(resize)(tensor->storage, totalSize+storageOffset, 0);
    }
  }
  else
  {
    tensor->nDimension = 0;    
  }
}

inline real* THTensor_(data)(THTensor *tensor)
{
  return tensor->storage->data+tensor->storageOffset;
}
    
inline real* THTensor_(data1d)(THTensor *tensor, long i0)
{
  return tensor->storage->data+tensor->storageOffset+i0*tensor->stride[0];
}

inline real* THTensor_(data2d)(THTensor *tensor, long i0, long i1)
{
  return tensor->storage->data+tensor->storageOffset+i0*tensor->stride[0]+i1*tensor->stride[1];
}

inline real* THTensor_(data3d)(THTensor *tensor, long i0, long i1, long i2)
{
  return tensor->storage->data+tensor->storageOffset+i0*tensor->stride[0]+i1*tensor->stride[1]+i2*tensor->stride[2];
}

inline real* THTensor_(data4d)(THTensor *tensor, long i0, long i1, long i2, long i3)
{
  return tensor->storage->data+tensor->storageOffset+i0*tensor->stride[0]+i1*tensor->stride[1]+i2*tensor->stride[2]+i3*tensor->stride[3];
}

inline real* THTensor_(selectPtr)(THTensor *tensor, int dimension, long sliceIndex)
{
  return tensor->storage->data+tensor->storageOffset+sliceIndex*tensor->stride[dimension];
}

void THTensor_(copy)(THTensor *tensor, THTensor *src)
{
  TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_p = *src_p;)
}

#define IMPLEMENT_THTensor_COPY(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
  TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_p = (real)(*src_p);) \
}

IMPLEMENT_THTensor_COPY(Byte, unsigned char)
IMPLEMENT_THTensor_COPY(Char, char)
IMPLEMENT_THTensor_COPY(Short, short)
IMPLEMENT_THTensor_COPY(Int, int)
IMPLEMENT_THTensor_COPY(Long, long)
IMPLEMENT_THTensor_COPY(Float, float)
IMPLEMENT_THTensor_COPY(Double, double)

#endif
