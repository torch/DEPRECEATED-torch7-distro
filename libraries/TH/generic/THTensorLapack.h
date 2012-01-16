#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorLapack.h"
#else

TH_API void THTensor_(gesv)(THTensor *a_, THTensor *b_);
TH_API void THTensor_(gels)(THTensor *a_, THTensor *b_);
TH_API void THTensor_(syev)(THTensor *a_, THTensor *w_, const char *jobz, const char *uplo);
TH_API void THTensor_(gesvd)(THTensor *a_, THTensor *s_, THTensor *u_, THTensor *vt_, char jobu);

#endif
