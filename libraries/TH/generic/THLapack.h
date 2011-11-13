#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLapack.h"
#else



/* AX=B */
void THLapack_(gesv)(int n, int nrhs, real *a, int lda, int *ipiv, real *b, int ldb, int* info);
/* ||AX-B|| */
/* void THLapack_(gels)(char trans, long m, long n, real alpha, real *a, long lda, real *x, long incx, real beta, real *y, long incy); */

/* /\* eigenvalues, svd *\/ */
/* void THLapack_(gesvd)(char transa, char transb, long m, long n, long k, real alpha, real *a, long lda, real *b, long ldb, real beta, real *c, long ldc); */
/* void THLapack_(syev)(char transa, char transb, long m, long n, long k, real alpha, real *a, long lda, real *b, long ldb, real beta, real *c, long ldc); */

#endif
