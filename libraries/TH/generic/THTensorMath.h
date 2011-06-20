#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMath.h"
#else

void THTensor_(fill)(THTensor *self, real value);
void THTensor_(zero)(THTensor *self);

void THTensor_(add)(THTensor *self, real value);
void THTensor_(mul)(THTensor *self, real value);
void THTensor_(div)(THTensor *self, real value);

void THTensor_(cadd)(THTensor *self, real value, THTensor *src);  
void THTensor_(cmul)(THTensor *self, THTensor *src);
void THTensor_(cdiv)(THTensor *self, THTensor *src);

void THTensor_(addcmul)(THTensor *self, real value, THTensor *src1, THTensor *src2);
void THTensor_(addcdiv)(THTensor *self, real value, THTensor *src1, THTensor *src2);

real THTensor_(dot)(THTensor *self, THTensor *src);
  
real THTensor_(min)(THTensor *self);
real THTensor_(max)(THTensor *self);
real THTensor_(sum)(THTensor *self);

void THTensor_(addmv)(THTensor *self, real alpha, THTensor *mat, THTensor *vec);
void THTensor_(addmm)(THTensor *self, real alpha, THTensor *mat1, THTensor *mat2);
void THTensor_(addr)(THTensor *self, real alpha, THTensor *vec1, THTensor *vec2);

THTensor *THTensor_(newconv2_valid)(THTensor *image, THTensor *kernel, long srow, long scol);
void THTensor_(conv2_valid)(THTensor *self, THTensor *image, THTensor *kernel, long srow, long scol);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
void THTensor_(log)(THTensor *self);
void THTensor_(log1p)(THTensor *self);
void THTensor_(exp)(THTensor *self);
void THTensor_(cos)(THTensor *self);
void THTensor_(acos)(THTensor *self);
void THTensor_(cosh)(THTensor *self);
void THTensor_(sin)(THTensor *self);
void THTensor_(asin)(THTensor *self);
void THTensor_(sinh)(THTensor *self);
void THTensor_(tan)(THTensor *self);
void THTensor_(atan)(THTensor *self);
void THTensor_(tanh)(THTensor *self);
void THTensor_(pow)(THTensor *self, real value);
void THTensor_(sqrt)(THTensor *self);
void THTensor_(ceil)(THTensor *self);
void THTensor_(floor)(THTensor *self);
void THTensor_(abs)(THTensor *self);

real THTensor_(mean)(THTensor *self);
real THTensor_(var)(THTensor *self);
real THTensor_(std)(THTensor *self);
real THTensor_(norm)(THTensor *self, real value);
real THTensor_(dist)(THTensor *self, THTensor *src, real value);
#endif

#endif
