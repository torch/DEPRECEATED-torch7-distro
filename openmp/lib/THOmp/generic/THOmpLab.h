#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THOmpLab.h"
#else

void THOmpLab_(conv2DRevger)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol);
void THOmpLab_(conv2Dger)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char* type);
void THOmpLab_(conv2Dmv)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char *type);

#endif


