#include "THCTensorConv.h"
#include "THCGeneral.h"

/* 
   3D input, 4D kernel, 3D output
   matrix vector product like
   y <- Ax + beta*y
*/
TH_API void THCudaTensor_conv2Dmv(THCudaTensor *output, float beta, THCudaTensor *input,
                                  THCudaTensor *kernel, long srow, long scol, const char *type)
{

}

/* 
   3D input, 3D kernel, 4D output
   like rank1 update
   A <- xx' + beta*A
*/
TH_API void THCudaTensor_conv2Dger(THCudaTensor *output, float beta, THCudaTensor *input, 
                                   THCudaTensor *kernel, long srow, long scol, const char *type)
{

}

/* 
   3D input, 3D kernel, 4D output
   like rank1 update
   A <- xx' + beta*A
   for sr,sc=1 this is equivalent to xcorr2Dger, but otherwise it is useful for
   calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
*/
TH_API void THCudaTensor_conv2DRevger(THCudaTensor *output, float beta, THCudaTensor *input, 
                                      THCudaTensor *kernel, long srow, long scol)
{

}
