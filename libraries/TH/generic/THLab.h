#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLab.h"
#else

TH_API void THLab_(fill)(THTensor *r_, real value);
TH_API void THLab_(zero)(THTensor *r_);

TH_API accreal THLab_(dot)(THTensor *t, THTensor *src);
  
TH_API real THLab_(minall)(THTensor *t);
TH_API real THLab_(maxall)(THTensor *t);
TH_API accreal THLab_(sumall)(THTensor *t);

TH_API void THLab_(add)(THTensor *r_, THTensor *t, real value);
TH_API void THLab_(mul)(THTensor *r_, THTensor *t, real value);
TH_API void THLab_(div)(THTensor *r_, THTensor *t, real value);

TH_API void THLab_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src);  
TH_API void THLab_(cmul)(THTensor *r_, THTensor *t, THTensor *src);
TH_API void THLab_(cdiv)(THTensor *r_, THTensor *t, THTensor *src);

TH_API void THLab_(addcmul)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
TH_API void THLab_(addcdiv)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);

TH_API void THLab_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat,  THTensor *vec);
TH_API void THLab_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat1, THTensor *mat2);
TH_API void THLab_(addr)(THTensor *r_,  real beta, THTensor *t, real alpha, THTensor *vec1, THTensor *vec2);

TH_API long THLab_(numel)(THTensor *t);
TH_API void THLab_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension);
TH_API void THLab_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension);
TH_API void THLab_(sum)(THTensor *r_, THTensor *t, int dimension);
TH_API void THLab_(prod)(THTensor *r_, THTensor *t, int dimension);
TH_API void THLab_(cumsum)(THTensor *r_, THTensor *t, int dimension);
TH_API void THLab_(cumprod)(THTensor *r_, THTensor *t, int dimension);
TH_API accreal THLab_(trace)(THTensor *t);
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
TH_API accreal THLab_(norm)(THTensor *t, real value);
TH_API accreal THLab_(dist)(THTensor *a, THTensor *b, real value);

TH_API accreal THLab_(meanall)(THTensor *self);
TH_API accreal THLab_(varall)(THTensor *self);
TH_API accreal THLab_(stdall)(THTensor *self);

TH_API void THLab_(linspace)(THTensor *r_, real a, real b, long n);
TH_API void THLab_(logspace)(THTensor *r_, real a, real b, long n);
TH_API void THLab_(rand)(THTensor *r_, THLongStorage *size);
TH_API void THLab_(randn)(THTensor *r_, THLongStorage *size);

#endif

#endif
