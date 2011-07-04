#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLabConv.h"
#else

void THLab_(xcorr2DRevger)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol);
void THLab_(xcorr2Dger)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char* type);
void THLab_(xcorr2Dmv)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char *type);
void THLab_(xcorr2Dmul)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char *type);

#endif
