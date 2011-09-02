#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THVector.h"
#else

TH_API inline void THVector_(fill)(real *x, real c, long n);
TH_API inline void THVector_(add)(real *y, const real *x, const real c, const long n);
TH_API inline void THVector_(diff)(real *z, const real *x, const real *y, const long n);
TH_API inline void THVector_(scale)(real *y, const real c, const long n);
TH_API inline void THVector_(mul)(real *y, const real *x, const long n);

#endif
