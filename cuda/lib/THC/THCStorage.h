#ifndef THC_STORAGE_INC
#define THC_STORAGE_INC

#include "THStorage.h"

#define real float
#define Real Cuda

#define TH_GENERIC_FILE "generic/THStorage.h"
#include "generic/THStorage.h"
#undef TH_GENERIC_FILE

#undef real
#undef Real

void THFloatStorage_copyCuda(THFloatStorage *self, struct THCudaStorage *src);

#endif
