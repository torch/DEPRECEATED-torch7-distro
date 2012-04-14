#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorFftw.h"
#else

TH_API void THTensor_(fft)(THTensor *r_, THTensor *x_, long n);
TH_API void THTensor_(ifft)(THTensor *r_, THTensor *x_, long n);
/*
TH_API void THTensor_(fft2)(THTensor *r_, THTensor *x_, long m, long n);
TH_API void THTensor_(ifft2)(THTensor *r_, THTensor *x_, long m, long n);

TH_API void THTensor_(fftn)(THTensor *r_, THTensor *x_);
TH_API void THTensor_(ifftn)(THTensor *r_, THTensor *x_);
*/
#endif
