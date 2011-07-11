#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLab.h"
#else

TH_API void THLab_(add)(THTensor *r_, THTensor *t, real value);
TH_API void THLab_(mul)(THTensor *r_, THTensor *t, real value);
TH_API void THLab_(div)(THTensor *r_, THTensor *t, real value);

TH_API void THLab_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src);  
TH_API void THLab_(cmul)(THTensor *r_, THTensor *t, THTensor *src);
TH_API void THLab_(cdiv)(THTensor *r_, THTensor *t, THTensor *src);

TH_API void THLab_(numel)(long *n_, THTensor *t);
TH_API void THLab_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension);
TH_API void THLab_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension);
TH_API void THLab_(sum)(THTensor *r_, THTensor *t, int dimension);
TH_API void THLab_(prod)(THTensor *r_, THTensor *t, int dimension);
TH_API void THLab_(cumsum)(THTensor *r_, THTensor *t, int dimension);
TH_API void THLab_(cumprod)(THTensor *r_, THTensor *t, int dimension);
TH_API void THLab_(trace)(real *trace_, THTensor *t);
TH_API void THLab_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);

TH_API void THLab_(zeros)(THTensor *r_, THLongStorage *size);
TH_API void THLab_(ones)(THTensor *r_, THLongStorage *size);
TH_API void THLab_(diag)(THTensor *r_, THTensor *t, int k);
TH_API void THLab_(eye)(THTensor *r_, long n, long m);
TH_API void THLab_(range)(THTensor *r_, real xmin, real xmax, real step);
TH_API void THLab_(randperm)(THTensor *r_, long n);

TH_API void THLab_(reshape)(THTensor *r_, THTensor *t, THLongStorage *size);
TH_API void THLab_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder);
TH_API void THLab_(tril)(THTensor *r_, THTensor *t, long k);
TH_API void THLab_(triu)(THTensor *r_, THTensor *t, long k);
TH_API void THLab_(cat)(THTensor *r_, THTensor *ta, THTensor *tb, int dimension);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

TH_API void THLab_(log)(THTensor *r_, THTensor *t);
TH_API void THLab_(log1p)(THTensor *r_, THTensor *t);
TH_API void THLab_(exp)(THTensor *r_, THTensor *t);
TH_API void THLab_(cos)(THTensor *r_, THTensor *t);
TH_API void THLab_(acos)(THTensor *r_, THTensor *t);
TH_API void THLab_(cosh)(THTensor *r_, THTensor *t);
TH_API void THLab_(sin)(THTensor *r_, THTensor *t);
TH_API void THLab_(asin)(THTensor *r_, THTensor *t);
TH_API void THLab_(sinh)(THTensor *r_, THTensor *t);
TH_API void THLab_(tan)(THTensor *r_, THTensor *t);
TH_API void THLab_(atan)(THTensor *r_, THTensor *t);
TH_API void THLab_(tanh)(THTensor *r_, THTensor *t);
TH_API void THLab_(pow)(THTensor *r_, THTensor *t, real value);
TH_API void THLab_(sqrt)(THTensor *r_, THTensor *t);
TH_API void THLab_(ceil)(THTensor *r_, THTensor *t);
TH_API void THLab_(floor)(THTensor *r_, THTensor *t);
TH_API void THLab_(abs)(THTensor *r_, THTensor *t);

TH_API void THLab_(mean)(THTensor *r_, THTensor *t, int dimension);
TH_API void THLab_(std)(THTensor *r_, THTensor *t, int dimension, int flag);
TH_API void THLab_(var)(THTensor *r_, THTensor *t, int dimension, int flag);
TH_API void THLab_(norm)(real *norm_, THTensor *t, real value);
TH_API void THLab_(dist)(real *dist_, THTensor *a, THTensor *b, real value);

TH_API void THLab_(linspace)(THTensor *r_, real a, real b, long n);
TH_API void THLab_(logspace)(THTensor *r_, real a, real b, long n);
TH_API void THLab_(rand)(THTensor *r_, THLongStorage *size);
TH_API void THLab_(randn)(THTensor *r_, THLongStorage *size);

#endif

#endif
