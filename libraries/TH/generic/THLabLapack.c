#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLabLapack.c"
#else

TH_API void THLab_(gesv)(THTensor *x_, THTensor *a_, THTensor *b_)
{
  int n, nrhs, lda, ldb, info;
  THIntTensor *ipiv;

  THArgCheck(b_->nDimension == x_->nDimension, 1, "B and X should have same # if dims");
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

  if (b_ != x_)
  {
    THTensor_(resizeAs)(x_,b_);
    THTensor_(copy(x_,b_));
  }

  ipiv = THIntTensor_newWithSize1d((long)n);
  THLapack_(gesv)(n, nrhs, 
		  THTensor_(data)(a_), lda, THIntTensor_data(ipiv), 
		  THTensor_(data)(x_), ldb, &info);

  if (info < 0)
  {
    THError("Lapack gesv : Argument %d : illegal value", -info);
  }
  else if (info > 0)
  {
    THError("Lapack gesv : U(%d,%d) is zero, singular U.", info,info);
  }
  THIntTensor_free(ipiv);
}

#endif
