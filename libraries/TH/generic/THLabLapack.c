#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLabLapack.c"
#else

#define __lapackmax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define __lapackmin( a, b ) ( ((a) < (b)) ? (a) : (b) )

TH_API void THLab_(gesv)(THTensor *a_, THTensor *b_)
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

TH_API void THLab_(gels)(THTensor *a_, THTensor *b_)
{
  int m, n, nrhs, lda, ldb, info, lwork;
  char transpose;
  THTensor *A, *B;
  THTensor *work;
  
  THArgCheck(a_->nDimension == 2, 2, "A should be 2 dimensional");

  A = THTensor_(newContiguous)(a_);
  B = THTensor_(newContiguous)(b_);
  m = A->size[1];
  n = A->size[0];
  lda = n;
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

  lwork = __lapackmin(m,n) + __lapackmax(__lapackmax(m,n),nrhs)*64;
  
  work = THTensor_(newWithSize1d)(lwork);
  
  THLapack_(gels)('N', m, n, nrhs, THTensor_(data)(A), lda, 
		  THTensor_(data)(B), ldb, 
		  THTensor_(data)(work), lwork, &info);

  if (info != 0)
  {
    THError("Lapack gels : Argument %d : illegal value", -info);
  }
  THTensor_(free)(A);
  THTensor_(free)(work);
}

#endif
