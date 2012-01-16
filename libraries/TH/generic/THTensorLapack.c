#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorLapack.c"
#else

TH_API void THTensor_(gesv)(THTensor *a_, THTensor *b_)
{
  int n, nrhs, lda, ldb, info;
  THIntTensor *ipiv;
  THTensor *A, *B;
  
  THArgCheck(a_->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(a_->size[0] == a_->size[1], 2, "A should be symmetric");

  n = (int)a_->size[1];
  lda = n;
  ldb = n;
  if (b_->nDimension == 1)
  {
    nrhs = 1;
    THArgCheck(n == b_->size[0], 1, "size incompatible A,b");
  }
  else
  {
    nrhs = b_->size[0];
    THArgCheck(n == b_->size[1], 1, "size incompatible A,b");
  }

  A = THTensor_(newContiguous)(a_);
  B = THTensor_(newContiguous)(b_);
  ipiv = THIntTensor_newWithSize1d((long)n);
  THLapack_(gesv)(n, nrhs, 
		  THTensor_(data)(A), lda, THIntTensor_data(ipiv),
		  THTensor_(data)(B), ldb, &info);

  if(!THTensor_(isContiguous)(b_))
  {
    THTensor_(copy)(b_,B);
  }

  if (info < 0)
  {
    THError("Lapack gesv : Argument %d : illegal value", -info);
  }
  else if (info > 0)
  {
    THError("Lapack gesv : U(%d,%d) is zero, singular U.", info,info);
  }

  THIntTensor_free(ipiv);
  THTensor_(free)(A);
  THTensor_(free)(B);
}

TH_API void THTensor_(gels)(THTensor *a_, THTensor *b_)
{
  int m, n, nrhs, lda, ldb, info, lwork;
  char transpose;
  THTensor *A, *B;
  THTensor *work;
  real wkopt;
  
  THArgCheck(a_->nDimension == 2, 2, "A should be 2 dimensional");

  A = THTensor_(newContiguous)(a_);
  B = THTensor_(newContiguous)(b_);
  m = A->size[1];
  n = A->size[0];
  lda = m;
  ldb = m;
  if (b_->nDimension == 1)
  {
    nrhs = 1;
    THArgCheck(m == b_->size[0], 1, "size incompatible A,b");
  }
  else
  {
    nrhs = b_->size[0];
    THArgCheck(m == b_->size[1], 1, "size incompatible A,b");
  }

  // get optimal workspace size
  THLapack_(gels)('N', m, n, nrhs, THTensor_(data)(A), lda, 
		  THTensor_(data)(B), ldb, 
		  &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);
  THLapack_(gels)('N', m, n, nrhs, THTensor_(data)(A), lda, 
		  THTensor_(data)(B), ldb, 
		  THTensor_(data)(work), lwork, &info);

  //printf("lwork = %d,%g\n",lwork,THTensor_(data)(work)[0]);
  if (info != 0)
  {
    THError("Lapack gels : Argument %d : illegal value", -info);
  }
  THTensor_(free)(A);
  THTensor_(free)(B);
  THTensor_(free)(work);
}

TH_API void THTensor_(syev)(THTensor *a_, THTensor *w_, const char *jobz, const char *uplo)
{
  int n, lda, lwork, info;
  THTensor *A;
  THTensor *work;
  real wkopt;

  THArgCheck(a_->nDimension == 2, 2, "A should be 2 dimensional");
  A = THTensor_(newContiguous)(a_);
  n = A->size[1];
  lda = n;
  THTensor_(resize1d)(w_,n);
  // get optimal workspace size
  THLapack_(syev)(jobz[0], uplo[0], n, THTensor_(data)(A), lda,
		  THTensor_(data)(w_), &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);
  THLapack_(syev)(jobz[0], uplo[0], n, THTensor_(data)(A), lda,
		  THTensor_(data)(w_), THTensor_(data)(work), lwork, &info);

  if (info > 0)
  {
    THError(" Lapack syev : Failed to converge. %d off-diagonal elements of an didn't converge to zero",info);
  }
  else if (info < 0)
  {
    THError("Lapack syev : Argument %d : illegal value", -info);
  }
  THTensor_(free)(A);
  THTensor_(free)(work);
}

TH_API void THTensor_(gesvd)(THTensor *a_, THTensor *s_, THTensor *u_, THTensor *vt_, char jobu)
{
  int k,m, n, lda, ldu, ldvt, lwork, info;
  THTensor *A, *work;
  real wkopt;
  char jobvt = jobu;

  THArgCheck(a_->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(jobu == 'A' || jobu == 'S',4, "jobu can be 'A' or 'S'");
  A = THTensor_(newContiguous)(a_);
  m = A->size[1];
  n = A->size[0];
  k = (m < n ? m : n);

  lda = m;
  ldu = m;
  ldvt = n;
  THTensor_(resize1d)(s_,k);
  THTensor_(resize2d)(vt_,n,ldvt);
  if (jobu == 'A')
  {
    THTensor_(resize2d)(u_,m,ldu);
  }
  else
  {
    THTensor_(resize2d)(u_,k,ldu);
  }
  THLapack_(gesvd)(jobu,jobvt,
		   m,n,THTensor_(data)(A),lda,
		   THTensor_(data)(s_),
		   THTensor_(data)(u_),
		   ldu,
		   THTensor_(data)(vt_), ldvt,
		   &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);
  THLapack_(gesvd)(jobu,jobvt,
		   m,n,THTensor_(data)(A),lda,
		   THTensor_(data)(s_),
		   THTensor_(data)(u_),
		   ldu,
		   THTensor_(data)(vt_), ldvt,
		   THTensor_(data)(work),lwork, &info);
  if (info > 0)
  {
    THError(" Lapack gesvd : %d superdiagonals failed to converge.",info);
  }
  else if (info < 0)
  {
    THError("Lapack gesvd : Argument %d : illegal value", -info);
  }
  THTensor_(free)(A);
  THTensor_(free)(work);
}

#endif
