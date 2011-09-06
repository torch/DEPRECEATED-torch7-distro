#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THOmpLabConv.h"
#else

/*
void THOmpLab_(validConv2Dptr)(real *r_, real *t_, long ir, long ic, real *k_, long kr, long kc, long sr, long sc);
void THOmpLab_(fullConv2Dptr)(real *r_, real *t_, long ir, long ic, real *k_, long kr, long kc, long sr, long sc);
void THOmpLab_(validConv2DRevptr)(real *r_, real *t_, long ir, long ic, real *k_, long kr, long kc, long sr, long sc);
*/

void THOmpLab_(conv2DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
void THOmpLab_(conv2Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char* type);
void THOmpLab_(conv2Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *type);

#endif


