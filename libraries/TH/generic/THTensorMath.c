#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMath.c"
#else


void THTensor_(mul)(THTensor *tensor, real value)
{
  /* we use a trick here. careful with that. */
  /* we do not have to increment stuff with this version (contrary to APPLY2 and APPLY3) */
  TH_TENSOR_APPLY(real, tensor, THBlas_(scal)(tensor_size, value, tensor_data, tensor_stride); break;);
}

void THTensor_(div)(THTensor *tensor, real value)
{
  THArgCheck(value != 0, 2, "division by 0");
  /* we use a trick here. careful with that. */
  /* we do not have to increment stuff with this version (contrary to APPLY2 and APPLY3) */
  TH_TENSOR_APPLY(real, tensor, THBlas_(scal)(tensor_size, 1/value, tensor_data, tensor_stride); break;);
}

void THTensor_(cadd)(THTensor *tensor, real value, THTensor *src)
{
  /* we use a trick here. careful with that. */
  TH_TENSOR_APPLY2(real, tensor, real, src,
                   long sz = (tensor_size-tensor_i < src_size-src_i ? tensor_size-tensor_i : src_size-src_i);
                   THBlas_(axpy)(sz, value, src_data, src_stride, tensor_data, tensor_stride);
                   tensor_i += sz;
                   src_i += sz;
                   tensor_data += sz*tensor_stride;
                   src_data += sz*src_stride;
                   break;);
}

void THTensor_(cmul)(THTensor *tensor, THTensor *src)
{
  TH_TENSOR_APPLY2(real, tensor, real, src,
                   long sz = (tensor_size-tensor_i < src_size-src_i ? tensor_size-tensor_i : src_size-src_i);
                   THVector_(mul)(tensor_data, src_data, sz);
                   tensor_i += sz;
                   src_i += sz;
                   tensor_data += sz*tensor_stride;
                   src_data += sz*src_stride;
                   break;);
}



#define TENSOR_IMPLEMENT_BASIC_FUNCTION(NAME, CFUNC)              \
  void THTensor_(NAME)(THTensor *tensor)                          \
  {                                                               \
    TH_TENSOR_APPLY(real, tensor, *tensor_data = CFUNC(*tensor_data);); \
  }

#define TENSOR_IMPLEMENT_BASIC_FUNCTION_VALUE(NAME, CFUNC)              \
  void THTensor_(NAME)(THTensor *tensor, real value)                    \
  {                                                                     \
    TH_TENSOR_APPLY(real, tensor, *tensor_data = CFUNC(*tensor_data, value);); \
  }

TENSOR_IMPLEMENT_BASIC_FUNCTION(log, log)
TENSOR_IMPLEMENT_BASIC_FUNCTION(log1p, log1p)
TENSOR_IMPLEMENT_BASIC_FUNCTION(exp, exp)
TENSOR_IMPLEMENT_BASIC_FUNCTION(cos, cos)
TENSOR_IMPLEMENT_BASIC_FUNCTION(acos, acos)
TENSOR_IMPLEMENT_BASIC_FUNCTION(cosh, cosh)
TENSOR_IMPLEMENT_BASIC_FUNCTION(sin, sin)
TENSOR_IMPLEMENT_BASIC_FUNCTION(asin, asin)
TENSOR_IMPLEMENT_BASIC_FUNCTION(sinh, sinh)
TENSOR_IMPLEMENT_BASIC_FUNCTION(tan, tan)
TENSOR_IMPLEMENT_BASIC_FUNCTION(atan, atan)
TENSOR_IMPLEMENT_BASIC_FUNCTION(tanh, tanh)
TENSOR_IMPLEMENT_BASIC_FUNCTION_VALUE(pow, pow)
TENSOR_IMPLEMENT_BASIC_FUNCTION(sqrt, sqrt)
TENSOR_IMPLEMENT_BASIC_FUNCTION(ceil, ceil)
TENSOR_IMPLEMENT_BASIC_FUNCTION(floor, floor)
TENSOR_IMPLEMENT_BASIC_FUNCTION(abs, fabs)


/* basic statistics */

 




#endif
