#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THVector.c"
#else

inline void THVector_(fill)(real *x, const real c, const long n) {
  long i;
  for(i=0; i<n; i++) {
    x[i] = c;
  }
}

inline void THVector_(add)(real *y, const real *x, const real c, const long n)
{
  long i;
  for(i=0; i<n; i++) {
    y[i] += c * x[i];
  }
}

inline void THVector_(diff)(real *z, const real *x, const real *y, const long n)
{
  long i;
  for(i=0; i<n; i++) {
    z[i] = x[i] - y[i];
  }
}

inline void THVector_(scale)(real *y, const real c, const long n)
{
  long i;
  for(i=0; i<n; i++) {
    y[i] *= c;
  }
}

inline void THVector_(mul)(real *y, const real *x, const long n)
{
  long i;
  for(i=0; i<n; i++) {
    y[i] *= x[i];
  }
}

#endif
