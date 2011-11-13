#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLapack.c"
#else

void THLapack_(gesv)(int n, int nrhs, real *a, int lda, int *ipiv, real *b, int ldb, int* info)
{
#if defined(TH_REAL_IS_DOUBLE)
  extern void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
  dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#else
  extern void sgesv_(int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info);
  sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#endif
  return;
}

#endif
