#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLabConv.h"
#else

TH_API void THLab_(conv2DRevger)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol);
TH_API void THLab_(conv2Dger)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char* type);
TH_API void THLab_(conv2Dmv)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char *type);
TH_API void THLab_(conv2Dmul)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char *type);

#endif
