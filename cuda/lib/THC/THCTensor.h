#ifndef THC_TENSOR_INC
#define THC_TENSOR_INC

#include "THTensor.h"
#include "THCStorage.h"

#define real float
#define Real Cuda
#define TH_GENERIC_FILE "generic/THTensor.h"

#include "generic/THTensor.h"

#undef TH_GENERIC_FILE
#undef real
#undef Real

#endif
