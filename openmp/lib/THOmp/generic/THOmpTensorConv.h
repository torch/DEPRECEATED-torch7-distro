#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THOmpTensorConv.h"
#else

/*
void THOmpTensor_(validConv2Dptr)(real *r_, real *t_, long ir, long ic, real *k_, long kr, long kc, long sr, long sc);
void THOmpTensor_(fullConv2Dptr)(real *r_, real *t_, long ir, long ic, real *k_, long kr, long kc, long sr, long sc);
void THOmpTensor_(validConv2DRevptr)(real *r_, real *t_, long ir, long ic, real *k_, long kr, long kc, long sr, long sc);
*/

void THOmpTensor_(conv2DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
void THOmpTensor_(conv2DRevgerm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol);
void THOmpTensor_(conv2Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char* type);
void THOmpTensor_(conv2Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *type);
void THOmpTensor_(conv2Dmm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, long srow, long scol, const char *type);

#endif


