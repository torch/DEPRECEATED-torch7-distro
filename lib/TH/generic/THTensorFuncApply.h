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

#endif
