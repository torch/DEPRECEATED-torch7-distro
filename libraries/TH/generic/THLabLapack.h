#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLabLapack.h"
#else

TH_API void THLab_(gesv)(THTensor *a_, THTensor *b_);
TH_API void THLab_(gels)(THTensor *a_, THTensor *b_);

#endif
