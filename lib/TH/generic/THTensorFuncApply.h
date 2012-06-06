#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorFuncApply.h"
#else

static void THTensor_(apply)(THTensor *a,
                             void (*func)(real *, void *),
                             void (*func_sz)(real *, long, void *),
                             void *data)
{
  if(func_sz && THTensor_(isContiguous)(a))
    func_sz(THTensor_(data)(a), THTensor_(nElement)(a), data);
  else
  {
    TH_TENSOR_APPLY(real, a, func(a_data, data);)
  }
}

static void THTensor_(apply2)(THTensor *a, THTensor *b,
                              void (*func)(real *, real *, void *),
                              void (*func_sz)(real *, real *, long, void *),
                              void *data)
{
  if(func_sz && THTensor_(isContiguous)(a) && THTensor_(isContiguous)(b))
    func_sz(THTensor_(data)(a), THTensor_(data)(b), THTensor_(nElement)(a), data);
  else
  {
    TH_TENSOR_APPLY2(real, a, real, b, func(a_data, b_data, data);)
  }
}

static void THTensor_(apply3)(THTensor *a, THTensor *b, THTensor *c,
                              void (*func)(real*, real*, real*, void*),
                              void (*func_sz)(real *, real *, real *, long, void *),
                              void *data)
{
  if(func_sz && THTensor_(isContiguous)(a) && THTensor_(isContiguous)(b) && THTensor_(isContiguous)(c))
    func_sz(THTensor_(data)(a), THTensor_(data)(b), THTensor_(data)(c), THTensor_(nElement)(a), data);
  else
  {
    TH_TENSOR_APPLY3(real, a, real, b, real, c, func(a_data, b_data, c_data, data);)
  }
}

static void THTensor_(dimapply2)(THTensor *a,
                                 THTensor *b,
                                 int dimension,
                                 void (*func)(real *, long, long, /* tensor, size, stride */
                                              real *, long, long, /* tensor, size, stride */
                                              void *), /* closure */
                                 void *data)
{
  if( a->nDimension != b->nDimension )
    THError("inconsistent tensor sizes");

  if( (dimension < 0) || (dimension >= a->nDimension) )
    THError("invalid dimension");

  real *a_data = THTensor_(data)(a);
  real *b_data = THTensor_(data)(b);
  int dim = a->nDimension;

  long n = 1;
  long i;
  for(i = 0; i < dim; i++)
  {
    if(i != dimension)
    {
      if(a->size[i] != b->size[i])
        THError("inconsistent tensor sizes");

      n *= a->size[i];
    }
  }

#pragma omp parallel for private(i)
  for(i = 0; i < n; i++)
  {
    real *a_data_ = a_data;
    real *b_data_ = b_data;
    long modulo = n;
    long rest = i;
    int d;
    for(d = 0; d < dim; d++)
    {
      if(d != dimension)
      {
        long idx;
        modulo = modulo / a->size[d];
        idx = rest / modulo;
        rest = rest % modulo;

        a_data_ += idx*a->stride[d];
        b_data_ += idx*b->stride[d];
      }
    }

    func(a_data_, a->size[dimension], a->stride[dimension],
         b_data_, b->size[dimension], b->stride[dimension],
         data);
  }
}

#endif
