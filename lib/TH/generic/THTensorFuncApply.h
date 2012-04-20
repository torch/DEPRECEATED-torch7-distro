#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorFuncApply.h"
#else

static void THTensor_(apply)(THTensor *tensor, void (*func)(real *))
{
  TH_TENSOR_APPLY(real, tensor, func(tensor_data);)
}

#endif
