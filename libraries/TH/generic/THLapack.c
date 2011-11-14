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

void THLapack_(gels)(char trans, int m, int n, int nrhs, real *a, int lda, real *b, int ldb, real *work, int lwork, int *info)
{
#if defined(TH_REAL_IS_DOUBLE)
  extern void dgels_(char *trans, int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *work, int *lwork, int *info);
  dgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
#else
  extern void sgels_(char *trans, int *m, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, float *work, int *lwork, int *info);
  sgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
#endif
}

#endif
