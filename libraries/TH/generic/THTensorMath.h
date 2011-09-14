#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMath.h"
#else

TH_API void THTensor_(fill)(THTensor *self, real value);
TH_API void THTensor_(zero)(THTensor *self);

TH_API void THTensor_(add)(THTensor *self, real value);
TH_API void THTensor_(mul)(THTensor *self, real value);
TH_API void THTensor_(div)(THTensor *self, real value);

TH_API void THTensor_(cadd)(THTensor *self, real value, THTensor *src);  
TH_API void THTensor_(cmul)(THTensor *self, THTensor *src);
TH_API void THTensor_(cdiv)(THTensor *self, THTensor *src);

TH_API void THTensor_(addcmul)(THTensor *self, real value, THTensor *src1, THTensor *src2);
TH_API void THTensor_(addcdiv)(THTensor *self, real value, THTensor *src1, THTensor *src2);

TH_API accreal THTensor_(dot)(THTensor *self, THTensor *src);
  
TH_API real THTensor_(min)(THTensor *self);
TH_API real THTensor_(max)(THTensor *self);
TH_API accreal THTensor_(sum)(THTensor *self);

TH_API void THTensor_(addmv)(THTensor *self, real beta, real alpha, THTensor *mat, THTensor *vec);
TH_API void THTensor_(addmm)(THTensor *self, real beta, real alpha, THTensor *mat1, THTensor *mat2);
TH_API void THTensor_(addr)(THTensor *self, real alpha, THTensor *vec1, THTensor *vec2);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
TH_API void THTensor_(log)(THTensor *self);
TH_API void THTensor_(log1p)(THTensor *self);
TH_API void THTensor_(exp)(THTensor *self);
TH_API void THTensor_(cos)(THTensor *self);
TH_API void THTensor_(acos)(THTensor *self);
TH_API void THTensor_(cosh)(THTensor *self);
TH_API void THTensor_(sin)(THTensor *self);
TH_API void THTensor_(asin)(THTensor *self);
TH_API void THTensor_(sinh)(THTensor *self);
TH_API void THTensor_(tan)(THTensor *self);
TH_API void THTensor_(atan)(THTensor *self);
TH_API void THTensor_(tanh)(THTensor *self);
TH_API void THTensor_(pow)(THTensor *self, real value);
TH_API void THTensor_(sqrt)(THTensor *self);
TH_API void THTensor_(ceil)(THTensor *self);
TH_API void THTensor_(floor)(THTensor *self);
TH_API void THTensor_(abs)(THTensor *self);

TH_API accreal THTensor_(mean)(THTensor *self);
TH_API accreal THTensor_(var)(THTensor *self);
TH_API accreal THTensor_(std)(THTensor *self);
TH_API accreal THTensor_(norm)(THTensor *self, real value);
TH_API accreal THTensor_(dist)(THTensor *self, THTensor *src, real value);
#endif

#endif
