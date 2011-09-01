#ifndef TH_VECTOR_INC
#define TH_VECTOR_INC

#include "THGeneral.h"

#define THVector_(NAME) TH_CONCAT_4(TH,Real,Vector_,NAME)

#if defined(__SSE2__)

#include <emmintrin.h>

#define THDoubleVector_fill(x, c, n) {          \
    int i;                                      \
    __m128d XMM0 = _mm_set1_pd(c);              \
    for (i=0; i<=((n)-8); i+=8) {               \
      _mm_store_pd((x)+i  , XMM0);              \
      _mm_store_pd((x)+i+2, XMM0);              \
      _mm_store_pd((x)+i+4, XMM0);              \
      _mm_store_pd((x)+i+6, XMM0);              \
    }                                           \
    int off = (n/8)*8;                          \
    for (i=0; i<=((n)%8); i++) {                \
      x[off+i] = c;                             \
    }                                           \
  }

#define THDoubleVector_add(y, x, c, n) {        \
    int i;                                      \
    __m128d XMM7 = _mm_set1_pd(c);              \
    for (i=0; i<=((n)-4); i+=4) {               \
      __m128d XMM0 = _mm_load_pd((x)+i  );      \
      __m128d XMM1 = _mm_load_pd((x)+i+2);      \
      __m128d XMM2 = _mm_load_pd((y)+i  );      \
      __m128d XMM3 = _mm_load_pd((y)+i+2);      \
      XMM0 = _mm_mul_pd(XMM0, XMM7);            \
      XMM1 = _mm_mul_pd(XMM1, XMM7);            \
      XMM2 = _mm_add_pd(XMM2, XMM0);            \
      XMM3 = _mm_add_pd(XMM3, XMM1);            \
      _mm_store_pd((y)+i  , XMM2);              \
      _mm_store_pd((y)+i+2, XMM3);              \
    }                                           \
    int off = (n/4)*4;                          \
    for (i=0; i<=((n)%4); i++) {                \
      y[i] += c * x[i];                         \
    }                                           \
  }

#define THDoubleVector_diff(z, x, y, n) {       \
    int i;                                      \
    for (i=0; i<=((n)-8); i+=8) {               \
      __m128d XMM0 = _mm_load_pd((x)+i  );      \
      __m128d XMM1 = _mm_load_pd((x)+i+2);      \
      __m128d XMM2 = _mm_load_pd((x)+i+4);      \
      __m128d XMM3 = _mm_load_pd((x)+i+6);      \
      __m128d XMM4 = _mm_load_pd((y)+i  );      \
      __m128d XMM5 = _mm_load_pd((y)+i+2);      \
      __m128d XMM6 = _mm_load_pd((y)+i+4);      \
      __m128d XMM7 = _mm_load_pd((y)+i+6);      \
      XMM0 = _mm_sub_pd(XMM0, XMM4);            \
      XMM1 = _mm_sub_pd(XMM1, XMM5);            \
      XMM2 = _mm_sub_pd(XMM2, XMM6);            \
      XMM3 = _mm_sub_pd(XMM3, XMM7);            \
      _mm_store_pd((z)+i  , XMM0);              \
      _mm_store_pd((z)+i+2, XMM1);              \
      _mm_store_pd((z)+i+4, XMM2);              \
      _mm_store_pd((z)+i+6, XMM3);              \
    }                                           \
    int off = (n/8)*8;                          \
    for (i=0; i<=((n)%8); i++) {                \
      z[i] = x[i] - y[i];                       \
    }                                           \
  }

#define THDoubleVector_scale(y, c, n) {         \
    int i;                                      \
    __m128d XMM7 = _mm_set1_pd(c);              \
    for (i=0; i<=((n)-4); i+=4) {               \
      __m128d XMM0 = _mm_load_pd((y)+i  );      \
      __m128d XMM1 = _mm_load_pd((y)+i+2);      \
      XMM0 = _mm_mul_pd(XMM0, XMM7);            \
      XMM1 = _mm_mul_pd(XMM1, XMM7);            \
      _mm_store_pd((y)+i  , XMM0);              \
      _mm_store_pd((y)+i+2, XMM1);              \
    }                                           \
    int off = (n/4)*4;                          \
    for (i=0; i<=((n)%4); i++) {                \
      y[i] *= c;                                \
    }                                           \
  }

#define THDoubleVector_mul(y, x, n) {           \
    int i;                                      \
    for (i=0; i<=((n)-8); i+=8) {               \
      __m128d XMM0 = _mm_load_pd((x)+i  );      \
      __m128d XMM1 = _mm_load_pd((x)+i+2);      \
      __m128d XMM2 = _mm_load_pd((x)+i+4);      \
      __m128d XMM3 = _mm_load_pd((x)+i+6);      \
      __m128d XMM4 = _mm_load_pd((y)+i  );      \
      __m128d XMM5 = _mm_load_pd((y)+i+2);      \
      __m128d XMM6 = _mm_load_pd((y)+i+4);      \
      __m128d XMM7 = _mm_load_pd((y)+i+6);      \
      XMM4 = _mm_mul_pd(XMM4, XMM0);            \
      XMM5 = _mm_mul_pd(XMM5, XMM1);            \
      XMM6 = _mm_mul_pd(XMM6, XMM2);            \
      XMM7 = _mm_mul_pd(XMM7, XMM3);            \
      _mm_store_pd((y)+i  , XMM4);              \
      _mm_store_pd((y)+i+2, XMM5);              \
      _mm_store_pd((y)+i+4, XMM6);              \
      _mm_store_pd((y)+i+6, XMM7);              \
    }                                           \
    int off = (n/8)*8;                          \
    for (i=0; i<=((n)%8); i++) {                \
      y[i] *= x[i];                             \
    }                                           \
  }

#define THFloatVector_fill(x, c, n) {           \
    int i;                                      \
    __m128 XMM0 = _mm_set_ps1(c);               \
    for (i=0; i<=((n)-16); i+=16) {             \
      _mm_store_ps((x)+i  ,  XMM0);             \
      _mm_store_ps((x)+i+4,  XMM0);             \
      _mm_store_ps((x)+i+8,  XMM0);             \
      _mm_store_ps((x)+i+12, XMM0);             \
    }                                           \
    int off = (n/16)*16;                        \
    for (i=0; i<=((n)%16); i++) {               \
      x[off+i] = c;                             \
    }                                           \
  }

#define THFloatVector_add(y, x, c, n) {         \
    int i;                                      \
    __m128 XMM7 = _mm_set_ps1(c);               \
    for (i=0; i<=((n)-8); i+=8) {               \
      __m128 XMM0 = _mm_load_ps((x)+i  );       \
      __m128 XMM1 = _mm_load_ps((x)+i+4);       \
      __m128 XMM2 = _mm_load_ps((y)+i  );       \
      __m128 XMM3 = _mm_load_ps((y)+i+4);       \
      XMM0 = _mm_mul_ps(XMM0, XMM7);            \
      XMM1 = _mm_mul_ps(XMM1, XMM7);            \
      XMM2 = _mm_add_ps(XMM2, XMM0);            \
      XMM3 = _mm_add_ps(XMM3, XMM1);            \
      _mm_store_ps((y)+i  , XMM2);              \
      _mm_store_ps((y)+i+4, XMM3);              \
    }                                           \
    int off = (n/8)*8;                          \
    for (i=0; i<=((n)%8); i++) {                \
      y[i] += c * x[i];                         \
    }                                           \
  }

#define THFloatVector_diff(z, x, y, n) {        \
    int i;                                      \
    for (i=0; i<=((n)-16); i+=16) {             \
      __m128 XMM0 = _mm_load_ps((x)+i   );      \
      __m128 XMM1 = _mm_load_ps((x)+i+ 4);      \
      __m128 XMM2 = _mm_load_ps((x)+i+ 8);      \
      __m128 XMM3 = _mm_load_ps((x)+i+12);      \
      __m128 XMM4 = _mm_load_ps((y)+i   );      \
      __m128 XMM5 = _mm_load_ps((y)+i+ 4);      \
      __m128 XMM6 = _mm_load_ps((y)+i+ 8);      \
      __m128 XMM7 = _mm_load_ps((y)+i+12);      \
      XMM0 = _mm_sub_ps(XMM0, XMM4);            \
      XMM1 = _mm_sub_ps(XMM1, XMM5);            \
      XMM2 = _mm_sub_ps(XMM2, XMM6);            \
      XMM3 = _mm_sub_ps(XMM3, XMM7);            \
      _mm_store_ps((z)+i   , XMM0);             \
      _mm_store_ps((z)+i+ 4, XMM1);             \
      _mm_store_ps((z)+i+ 8, XMM2);             \
      _mm_store_ps((z)+i+12, XMM3);             \
    }                                           \
    int off = (n/16)*16;                        \
    for (i=0; i<=((n)%16); i++) {               \
      z[i] = x[i] - y[i];                       \
    }                                           \
  }

#define THFloatVector_scale(y, c, n) {          \
    int i;                                      \
    __m128 XMM7 = _mm_set_ps1(c);               \
    for (i=0; i<=((n)-8); i+=8) {               \
      __m128 XMM0 = _mm_load_ps((y)+i  );       \
      __m128 XMM1 = _mm_load_ps((y)+i+4);       \
      XMM0 = _mm_mul_ps(XMM0, XMM7);            \
      XMM1 = _mm_mul_ps(XMM1, XMM7);            \
      _mm_store_ps((y)+i  , XMM0);              \
      _mm_store_ps((y)+i+4, XMM1);              \
    }                                           \
    int off = (n/8)*8;                          \
    for (i=0; i<=((n)%8); i++) {                \
      y[i] *= c;                                \
    }                                           \
  }

#define THFloatVector_mul(y, x, n) {            \
    int i;                                      \
    for (i=0; i<=((n)-16); i+=16) {             \
      __m128 XMM0 = _mm_load_ps((x)+i   );      \
      __m128 XMM1 = _mm_load_ps((x)+i+ 4);      \
      __m128 XMM2 = _mm_load_ps((x)+i+ 8);      \
      __m128 XMM3 = _mm_load_ps((x)+i+12);      \
      __m128 XMM4 = _mm_load_ps((y)+i   );      \
      __m128 XMM5 = _mm_load_ps((y)+i+ 4);      \
      __m128 XMM6 = _mm_load_ps((y)+i+ 8);      \
      __m128 XMM7 = _mm_load_ps((y)+i+12);      \
      XMM4 = _mm_mul_ps(XMM4, XMM0);            \
      XMM5 = _mm_mul_ps(XMM5, XMM1);            \
      XMM6 = _mm_mul_ps(XMM6, XMM2);            \
      XMM7 = _mm_mul_ps(XMM7, XMM3);            \
      _mm_store_ps((y)+i   , XMM4);             \
      _mm_store_ps((y)+i+ 4, XMM5);             \
      _mm_store_ps((y)+i+ 8, XMM6);             \
      _mm_store_ps((y)+i+12, XMM7);             \
    }                                           \
    int off = (n/16)*16;                        \
    for (i=0; i<=((n)%16); i++) {               \
      y[i] *= x[i];                             \
    }                                           \
  }

#else

// If SSE2 not defined, then generate plain C operators
#include "generic/THVector.h"
#include "THGenerateFloatTypes.h"

#endif

// For non-float types, generate plain C operators
#include "generic/THVector.h"
#include "THGenerateIntTypes.h"

#endif
