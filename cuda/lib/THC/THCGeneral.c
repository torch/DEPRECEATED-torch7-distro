#include "THCGeneral.h"
#include "TH.h"

void __THCudaCheck(cudaError_t err, const char *file, const int line)
{
  if( cudaSuccess != err)
  {
    THError("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
            file, line, cudaGetErrorString(err));
  }
}
