#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLab.h"
#else

void THLab_(numel)(long *n_, THTensor *t);
void THLab_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension);
void THLab_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension);
void THLab_(sum)(THTensor *r_, THTensor *t, int dimension);
void THLab_(prod)(THTensor *r_, THTensor *t, int dimension);
void THLab_(cumsum)(THTensor *r_, THTensor *t, int dimension);
void THLab_(cumprod)(THTensor *r_, THTensor *t, int dimension);
void THLab_(trace)(real *trace_, THTensor *t);
void THLab_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);

void THLab_(zeros)(THTensor *r_, THLongStorage *size);
void THLab_(ones)(THTensor *r_, THLongStorage *size);
void THLab_(diag)(THTensor *r_, THTensor *t, int k);
void THLab_(eye)(THTensor *r_, long n, long m);
void THLab_(range)(THTensor *r_, real xmin, real xmax, real step);
void THLab_(randperm)(THTensor *r_, long n);

void THLab_(reshape)(THTensor *r_, THTensor *t, THLongStorage *size);
void THLab_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder);
void THLab_(tril)(THTensor *r_, THTensor *t, long k);
void THLab_(triu)(THTensor *r_, THTensor *t, long k);
void THLab_(cat)(THTensor *r_, THTensor *ta, THTensor *tb, int dimension);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

void THLab_(log)(THTensor *r_, THTensor *t);
void THLab_(log1p)(THTensor *r_, THTensor *t);
void THLab_(exp)(THTensor *r_, THTensor *t);
void THLab_(cos)(THTensor *r_, THTensor *t);
void THLab_(acos)(THTensor *r_, THTensor *t);
void THLab_(cosh)(THTensor *r_, THTensor *t);
void THLab_(sin)(THTensor *r_, THTensor *t);
void THLab_(asin)(THTensor *r_, THTensor *t);
void THLab_(sinh)(THTensor *r_, THTensor *t);
void THLab_(tan)(THTensor *r_, THTensor *t);
void THLab_(atan)(THTensor *r_, THTensor *t);
void THLab_(tanh)(THTensor *r_, THTensor *t);
void THLab_(pow)(THTensor *r_, THTensor *t, real value);
void THLab_(sqrt)(THTensor *r_, THTensor *t);
void THLab_(ceil)(THTensor *r_, THTensor *t);
void THLab_(floor)(THTensor *r_, THTensor *t);
void THLab_(abs)(THTensor *r_, THTensor *t);

void THLab_(mean)(THTensor *r_, THTensor *t, int dimension);
void THLab_(std)(THTensor *r_, THTensor *t, int dimension, int flag);
void THLab_(var)(THTensor *r_, THTensor *t, int dimension, int flag);
void THLab_(norm)(real *norm_, THTensor *t, real value);
void THLab_(dist)(real *dist_, THTensor *a, THTensor *b, real value);

void THLab_(linspace)(THTensor *r_, real a, real b, long n);
void THLab_(logspace)(THTensor *r_, real a, real b, long n);
void THLab_(rand)(THTensor *r_, THLongStorage *size);
void THLab_(randn)(THTensor *r_, THLongStorage *size);

#endif

#endif
