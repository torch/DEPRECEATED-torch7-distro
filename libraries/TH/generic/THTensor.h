#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensor.h"
#else

/* a la lua? dim, storageoffset, ...  et les methodes ? */

typedef struct THTensor
{
    long *size;
    long *stride;
    int nDimension;
    
    THStorage *storage;
    long storageOffset;
    int refcount;

} THTensor;


/**** access methods ****/
TH_API THStorage* THTensor_(storage)(THTensor *self);
TH_API long THTensor_(storageOffset)(THTensor *self);
TH_API int THTensor_(nDimension)(THTensor *self);
TH_API long THTensor_(size)(THTensor *self, int dim);
TH_API long THTensor_(stride)(THTensor *self, int dim);
TH_API THLongStorage *THTensor_(newSizeOf)(THTensor *self);
TH_API THLongStorage *THTensor_(newStrideOf)(THTensor *self);
TH_API real *THTensor_(data)(THTensor *self);


/**** creation methods ****/
TH_API THTensor *THTensor_(new)(void);
TH_API THTensor *THTensor_(newWithTensor)(THTensor *tensor);
/* stride might be NULL */
TH_API THTensor *THTensor_(newWithStorage)(THStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
TH_API THTensor *THTensor_(newWithStorage1d)(THStorage *storage_, long storageOffset_,
                                long size0_, long stride0_);
TH_API THTensor *THTensor_(newWithStorage2d)(THStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
TH_API THTensor *THTensor_(newWithStorage3d)(THStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
TH_API THTensor *THTensor_(newWithStorage4d)(THStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);

/* stride might be NULL */
TH_API THTensor *THTensor_(newWithSize)(THLongStorage *size_, THLongStorage *stride_);
TH_API THTensor *THTensor_(newWithSize1d)(long size0_);
TH_API THTensor *THTensor_(newWithSize2d)(long size0_, long size1_);
TH_API THTensor *THTensor_(newWithSize3d)(long size0_, long size1_, long size2_);
TH_API THTensor *THTensor_(newWithSize4d)(long size0_, long size1_, long size2_, long size3_);

TH_API THTensor *THTensor_(newContiguous)(THTensor *tensor);
  
TH_API void THTensor_(resize)(THTensor *tensor, THLongStorage *size, THLongStorage *stride);
TH_API void THTensor_(resizeAs)(THTensor *tensor, THTensor *src);
TH_API void THTensor_(resize1d)(THTensor *tensor, long size0_);
TH_API void THTensor_(resize2d)(THTensor *tensor, long size0_, long size1_);
TH_API void THTensor_(resize3d)(THTensor *tensor, long size0_, long size1_, long size2_);
TH_API void THTensor_(resize4d)(THTensor *tensor, long size0_, long size1_, long size2_, long size3_);

TH_API void THTensor_(narrow)(THTensor *self, int dimension_, long firstIndex_, long size_);
TH_API void THTensor_(select)(THTensor *self, int dimension_, long sliceIndex_);
TH_API void THTensor_(transpose)(THTensor *self, int dimension1_, int dimension2_);
TH_API void THTensor_(unfold)(THTensor *self, int dimension_, long size_, long step_);
    
TH_API int THTensor_(isContiguous)(THTensor *self);
TH_API long THTensor_(nElement)(THTensor *self);

TH_API void THTensor_(retain)(THTensor *self);
TH_API void THTensor_(free)(THTensor *self);

/* Slow access methods [check everything] */
TH_API void THTensor_(set1d)(THTensor *tensor, long x0, real value);
TH_API void THTensor_(set2d)(THTensor *tensor, long x0, long x1, real value);
TH_API void THTensor_(set3d)(THTensor *tensor, long x0, long x1, long x2, real value);
TH_API void THTensor_(set4d)(THTensor *tensor, long x0, long x1, long x2, long x3, real value);

TH_API real THTensor_(get1d)(THTensor *tensor, long x0);
TH_API real THTensor_(get2d)(THTensor *tensor, long x0, long x1);
TH_API real THTensor_(get3d)(THTensor *tensor, long x0, long x1, long x2);
TH_API real THTensor_(get4d)(THTensor *tensor, long x0, long x1, long x2, long x3);

/* Support for copy between different Tensor types */

struct THByteTensor;
struct THCharTensor;
struct THShortTensor;
struct THIntTensor;
struct THLongTensor;
struct THFloatTensor;
struct THDoubleTensor;

TH_API void THTensor_(copy)(THTensor *tensor, THTensor *src);
TH_API void THTensor_(copyByte)(THTensor *tensor, struct THByteTensor *src);
TH_API void THTensor_(copyChar)(THTensor *tensor, struct THCharTensor *src);
TH_API void THTensor_(copyShort)(THTensor *tensor, struct THShortTensor *src);
TH_API void THTensor_(copyInt)(THTensor *tensor, struct THIntTensor *src);
TH_API void THTensor_(copyLong)(THTensor *tensor, struct THLongTensor *src);
TH_API void THTensor_(copyFloat)(THTensor *tensor, struct THFloatTensor *src);
TH_API void THTensor_(copyDouble)(THTensor *tensor, struct THDoubleTensor *src);

#endif
