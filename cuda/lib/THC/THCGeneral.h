#ifndef THC_GENERAL_INC
#define THC_GENERAL_INC

#include "cublas.h"

#define THCudaCheck(err)  __THCudaCheck(err, __FILE__, __LINE__)

void __THCudaCheck(cudaError_t err, const char *file, const int line);

#endif
