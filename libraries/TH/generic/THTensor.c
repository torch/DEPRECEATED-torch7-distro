#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensor.c"
#else

/**** access methods ****/
THStorage *THTensor_(storage)(THTensor *self)
{
  return self->storage;
}

long THTensor_(storageOffset)(THTensor *self)
{
  return self->storageOffset;
}

int THTensor_(nDimension)(THTensor *self)
{
  return self->nDimension;
}

long THTensor_(size)(THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->size[dim];
}

long THTensor_(stride)(THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->stride[dim];
}

THLongStorage *THTensor_(newSizeOf)(THTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongStorage *THTensor_(newStrideOf)(THTensor *self)
{
  THLongStorage *stride = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

real *THTensor_(data)(THTensor *self)
{
  if(self->storage)
    return (self->storage->data+self->storageOffset);
  else
    return NULL;
}

/**** creation methods ****/

static void THTensor_(rawInit)(THTensor *self, THStorage *storage, long storageOffset, int nDimension, long *size, long *stride);
static void THTensor_(rawResize)(THTensor *self, int nDimension, long *size, long *stride);


/* Empty init */
THTensor *THTensor_(new)(void)
{
  THTensor *self = THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self, NULL, 0, 0, NULL, NULL);
  return self;
}

/* Pointer-copy init */
THTensor *THTensor_(newWithTensor)(THTensor *tensor)
{
  THTensor *self = THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self, tensor->storage, tensor->storageOffset, tensor->nDimension, tensor->size, tensor->stride);
  return self;
}

/* Storage init */
THTensor *THTensor_(newWithStorage)(THStorage *storage, long storageOffset, THLongStorage *size, THLongStorage *stride)
{  
  THTensor *self = THAlloc(sizeof(THTensor));
  if(size && stride)
    THArgCheck(size->size == stride->size, 4, "inconsistent size");
  
  THTensor_(rawInit)(self, storage, storageOffset,
                  (size ? size->size : 0),
                  (size ? size->data : NULL),
                  (stride ? stride->data : NULL));
  return self;
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
  THLongStorage *size = THLongStorage_newWithSize4(size0, size1, size2, size3);
  THLongStorage *stride = THLongStorage_newWithSize4(stride0, stride1, stride2, stride3);
  THTensor *self = THTensor_(newWithStorage)(storage, storageOffset, size, stride);
  THLongStorage_free(size);
  THLongStorage_free(stride);
  return self;
}

THTensor *THTensor_(newWithSize)(int nDimension, THLongStorage *size, THLongStorage *stride)
{
  return THTensor_(newWithStorage)(NULL, 0, size, stride);
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
  THLongStorage *size = THLongStorage_newWithSize4(size0, size1, size2, size3);
  THTensor *self = THTensor_(newWithSize)(4, size, NULL);
  THLongStorage_free(size);
  return self;
}

THTensor *THTensor_(newContiguous)(THTensor *self)
{
  THTensor *tensor = THTensor_(new)();
  THTensor_(resizeAs)(tensor, self);
  THTensor_(copy)(tensor, self);
  return tensor;
}



/* Resize */
void THTensor_(resize)(THTensor *self, THLongStorage *size, THLongStorage *stride)
{
  THArgCheck(size != NULL, 2, "invalid size");
  if(stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

  THTensor_(rawResize)(self, size->size, size->data, (stride ? stride->data : NULL));
}

void THTensor_(resizeAs)(THTensor *self, THTensor *src)
{
  int isSame = 0;
  int d;
  if(self->nDimension == src->nDimension)
  {
    isSame = 1;
    for(d = 0; d < self->nDimension; d++)
    {
      if(self->size[d] != src->size[d])
      {
        isSame = 0;
        break;
      }
    }
  }

  if(!isSame)
    THTensor_(rawResize)(self, src->nDimension, src->size, NULL);
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

void THTensor_(resize4d)(THTensor *self, long size0, long size1, long size2, long size3)
{
  THLongStorage *size = THLongStorage_newWithSize4(size0, size1, size2, size3);
  THTensor_(resize)(self, size, NULL);
  THLongStorage_free(size);
}

void THTensor_(narrow)(THTensor *self, int dimension, long firstIndex, long size)
{
  THArgCheck( (dimension >= 0) && (dimension < self->nDimension), 2, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < self->size[dimension]), 3, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= self->size[dimension]), 4, "out of range");

  if(firstIndex > 0)
    self->storageOffset += firstIndex*self->stride[dimension];

  self->size[dimension] = size;
}

void THTensor_(select)(THTensor *self, int dimension, long sliceIndex)
{
  int d;

  THArgCheck(self->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < self->nDimension), 2, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < self->size[dimension]), 3, "out of range");

  THTensor_(narrow)(self, dimension, sliceIndex, 1);
  for(d = dimension; d < self->nDimension-1; d++)
  {
    self->size[d] = self->size[d+1];
    self->stride[d] = self->stride[d+1];
  }
  self->nDimension--;
}

void THTensor_(transpose)(THTensor *self, int dimension1, int dimension2)
{
  long z;

  THArgCheck( (dimension1 >= 0) && (dimension1 < self->nDimension), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < self->nDimension), 2, "out of range");

  if(dimension1 == dimension2)
	  return;
 
  z = self->stride[dimension1];
  self->stride[dimension1] = self->stride[dimension2];
  self->stride[dimension2] = z;
  z = self->size[dimension1];
  self->size[dimension1] = self->size[dimension2];
  self->size[dimension2] = z;
}

void THTensor_(unfold)(THTensor *self, int dimension, long size, long step)
{
  long *newSize;
  long *newStride;
  int d;

  THArgCheck( (self->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < self->nDimension, 2, "out of range");
  THArgCheck(size <= self->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  newSize = THAlloc(sizeof(long)*(self->nDimension+1));
  newStride = THAlloc(sizeof(long)*(self->nDimension+1));

  newSize[self->nDimension] = size;
  newStride[self->nDimension] = self->stride[dimension];
  for(d = 0; d < self->nDimension; d++)
  {
    if(d == dimension)
    {
      newSize[d] = (self->size[d] - size) / step + 1;
      newStride[d] = step*self->stride[d];
    }
    else
    {
      newSize[d] = self->size[d];
      newStride[d] = self->stride[d];
    }
  }

  THFree(self->size);
  THFree(self->stride);

  self->size = newSize;
  self->stride = newStride;
}

int THTensor_(isContiguous)(THTensor *self)
{
  long z = 1;
  int d;
  for(d = 0; d < self->nDimension; d++)
  {
    if(self->stride[d] == z)
      z *= self->size[d];
    else
      return 0;
  }
  return 1;
}

long THTensor_(nElement)(THTensor *self)
{
  if(self->nDimension == 0)
    return 0;
  else
  {
    long nElement = 1;
    int d;
    for(d = 0; d < self->nDimension; d++)
      nElement *= self->size[d];
    return nElement;
  }
}

void THTensor_(retain)(THTensor *self)
{
  ++self->refcount;
}

void THTensor_(free)(THTensor *self)
{
  if(!self)
    return;

  if(--self->refcount == 0)
  {
    THFree(self->size);
    THFree(self->stride);
    THStorage_(free)(self->storage);
    THFree(self);
  }
}

/*******************************************************************************/

/* This one does everything except coffee */
static void THTensor_(rawInit)(THTensor *self, THStorage *storage, long storageOffset, int nDimension, long *size, long *stride)
{
  /* storage */
  if(storage)
  {
    self->storage = storage;
    THStorage_(retain)(self->storage);
  }
  else
    self->storage = THStorage_(new)();

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
  self->storageOffset = storageOffset;


  /* nDimension, size and stride */
  self->size = NULL;
  self->stride = NULL;
  self->nDimension = 0;    

  THTensor_(rawResize)(self, nDimension, size, stride);
}

static void THTensor_(rawResize)(THTensor *self, int nDimension, long *size, long *stride)
{
  int d;
  int nDimension_;
  long totalSize;

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
    if(nDimension != self->nDimension)
    {
      self->size = THRealloc(self->size, sizeof(long)*nDimension);
      self->stride = THRealloc(self->stride, sizeof(long)*nDimension);
      self->nDimension = nDimension;
    }
  
    totalSize = 1;
    for(d = 0; d < self->nDimension; d++)
    {
      self->size[d] = size[d];
      if(stride && (stride[d] >= 0) )
        self->stride[d] = stride[d];
      else
      {
        if(d == 0)
          self->stride[d] = 1;
        else
          self->stride[d] = self->size[d-1]*self->stride[d-1];
      }
      totalSize += (self->size[d]-1)*self->stride[d];
    }
    
    if(totalSize+self->storageOffset > self->storage->size)
      THStorage_(resize)(self->storage, totalSize+self->storageOffset);
  }
  else
    self->nDimension = 0;
}

void THTensor_(copy)(THTensor *tensor, THTensor *src)
{
  TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = *src_data;)
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

#define IMPLEMENT_THTensor_COPY(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
  TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = (real)(*src_data);) \
}

IMPLEMENT_THTensor_COPY(Byte, unsigned char)
IMPLEMENT_THTensor_COPY(Char, char)
IMPLEMENT_THTensor_COPY(Short, short)
IMPLEMENT_THTensor_COPY(Int, int)
IMPLEMENT_THTensor_COPY(Long, long)
IMPLEMENT_THTensor_COPY(Float, float)
IMPLEMENT_THTensor_COPY(Double, double)

#endif
