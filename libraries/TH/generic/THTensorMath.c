#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMath.c"
#else

void THTensor_(fill)(THTensor *tensor, real value)
{
  TH_TENSOR_APPLY(real, tensor, 
                  THVector_(fill)(tensor_data, value, tensor_size); break;);
}

void THTensor_(zero)(THTensor *tensor)
{
  TH_TENSOR_APPLY(real, tensor, 
                  THVector_(fill)(tensor_data, 0, tensor_size); break;);
}

void THTensor_(add)(THTensor *tensor, real value)
{
  TH_TENSOR_APPLY(real, tensor, *tensor_data += value;);
}

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

void THTensor_(cdiv)(THTensor *tensor, THTensor *src)
{
  TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data /= *src_data;);
}

void THTensor_(addcmul)(THTensor *tensor, real value, THTensor *src1, THTensor *src2)
{
  TH_TENSOR_APPLY3(real, tensor, real, src1, real, src2, *tensor_data += value * *src1_data * *src2_data;);
}


void THTensor_(addcdiv)(THTensor *tensor, real value, THTensor *src1, THTensor *src2)
{
  TH_TENSOR_APPLY3(real, tensor, real, src1, real, src2, *tensor_data += value * *src1_data / *src2_data;);
}

accreal THTensor_(dot)(THTensor *tensor, THTensor *src)
{
  accreal sum = 0;
  /* we use a trick here. careful with that. */
  TH_TENSOR_APPLY2(real, tensor, real, src,
                   long sz = (tensor_size-tensor_i < src_size-src_i ? tensor_size-tensor_i : src_size-src_i);
                   sum += THBlas_(dot)(sz, src_data, src_stride, tensor_data, tensor_stride);
                   tensor_i += sz;
                   src_i += sz;
                   tensor_data += sz*tensor_stride;
                   src_data += sz*src_stride; 
                   break;);
  return sum; 
}

real THTensor_(min)(THTensor *tensor)
{
  real theMin;
  THArgCheck(tensor->nDimension > 0, 1, "tensor must have one dimension");
  theMin = THTensor_(data)(tensor)[0];
  TH_TENSOR_APPLY(real, tensor, if(*tensor_data < theMin) theMin = *tensor_data;);
  return theMin; 
}

real THTensor_(max)(THTensor *tensor)
{
  real theMax;
  THArgCheck(tensor->nDimension > 0, 1, "tensor must have one dimension");
  theMax = THTensor_(data)(tensor)[0];
  TH_TENSOR_APPLY(real, tensor, if(*tensor_data > theMax) theMax = *tensor_data;);
  return theMax; 
}

accreal THTensor_(sum)(THTensor *tensor)
{
  accreal sum = 0;
  TH_TENSOR_APPLY(real, tensor, sum += *tensor_data;);
  return sum;
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
accreal THTensor_(mean)(THTensor *tensor)
{ 
  THArgCheck(tensor->nDimension > 0, 1, "empty Tensor");
  return THTensor_(sum)(tensor)/THTensor_(nElement)(tensor);
}  

accreal THTensor_(var)(THTensor *tensor)
{ 
  accreal mean = THTensor_(mean)(tensor);
  accreal sum = 0;
  TH_TENSOR_APPLY(real, tensor, sum += (*tensor_data - mean)*(*tensor_data - mean););
  sum /= (THTensor_(nElement)(tensor)-1);
  return sum;
}

accreal THTensor_(std)(THTensor *tensor)
{ 
  return sqrt(THTensor_(var)(tensor));
} 
 
accreal THTensor_(norm)(THTensor *tensor, real value)
{ 
  accreal sum = 0;
  TH_TENSOR_APPLY(real, tensor, sum += pow(fabs(*tensor_data), value););
  return pow(sum, 1.0/value);
}

accreal THTensor_(dist)(THTensor *tensor, THTensor *src, real value)
{ 
  real sum = 0;
  TH_TENSOR_APPLY2(real, tensor, real, src, 
	sum += pow(fabs(*tensor_data - *src_data), value);)
  return pow(sum, 1.0/value);
}

void THTensor_(addmv)(THTensor *tensor, real beta, real alpha, THTensor *mat, THTensor *vec)
{
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected");
 
  if( mat->size[1] != vec->size[0] )
    THError("size mismatch");

  if(tensor->nDimension != 1)
    THError("size mismatch");
    
  if( tensor->size[0] != mat->size[0] )
    THError("size mismatch");

  if(mat->stride[0] == 1)
  {
    THBlas_(gemv)('n', mat->size[0], mat->size[1],
                  alpha, THTensor_(data)(mat), mat->stride[1],
                  THTensor_(data)(vec), vec->stride[0],
                  beta, THTensor_(data)(tensor), tensor->stride[0]);
  }
  else if(mat->stride[1] == 1)
  {
    THBlas_(gemv)('t',  mat->size[1], mat->size[0],
                  alpha, THTensor_(data)(mat), mat->stride[0],
                  THTensor_(data)(vec), vec->stride[0],
                  beta, THTensor_(data)(tensor), tensor->stride[0]);
  }
  else
  {
    THTensor *cmat = THTensor_(newContiguous)(mat);

    THBlas_(gemv)('t',  mat->size[1], mat->size[0],
                  alpha, THTensor_(data)(cmat), cmat->stride[0],
                  THTensor_(data)(vec), vec->stride[0],
                  beta, THTensor_(data)(tensor), tensor->stride[0]);

    THTensor_(free)(cmat);
  }
}

void THTensor_(addr)(THTensor *tensor, real alpha, THTensor *vec1, THTensor *vec2)
{
  if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected");

  if(tensor->nDimension != 2)
    THError("size mismatch");
    
  if( (tensor->size[0] != vec1->size[0]) || (tensor->size[1] != vec2->size[0]) )
    THError("size mismatch");

  if(tensor->stride[0] == 1)
  {
    THBlas_(ger)(vec1->size[0], vec2->size[0],
                 alpha, THTensor_(data)(vec1), vec1->stride[0],
                 THTensor_(data)(vec2), vec2->stride[0],
                 THTensor_(data)(tensor), tensor->stride[1]);
  }
  else if(tensor->stride[1] == 1)
  {
    THBlas_(ger)(vec2->size[0], vec1->size[0],
                 alpha, THTensor_(data)(vec2), vec2->stride[0],
                 THTensor_(data)(vec1), vec1->stride[0],
                 THTensor_(data)(tensor), tensor->stride[0]);
  }
  else
  {
    THTensor *ctensor = THTensor_(newClone)(tensor);

    THBlas_(ger)(vec2->size[0], vec1->size[0],
                 alpha, THTensor_(data)(vec2), vec2->stride[0],
                 THTensor_(data)(vec1), vec1->stride[0],
                 THTensor_(data)(ctensor), ctensor->stride[0]);

    THTensor_(freeCopyTo)(ctensor, tensor);
  }
}

void THTensor_(addmm)(THTensor *tensor, real beta, real alpha, THTensor *m1, THTensor *m2)
{ 
  long r, c;
  char transpose, transpose_m1, transpose_m2;
  THTensor *tensor_, *m1_, *m2_;

  if( (m1->nDimension != 2) || (m2->nDimension != 2) ) 
    THError("matrix and matrix expected"); 
 
  if(tensor->nDimension != 2)
    THError("size mismatch"); 

  if( (tensor->size[0] != m1->size[0]) || (tensor->size[1] != m2->size[1]) || (m1->size[1] != m2->size[0]) ) 
    THError("size mismatch"); 

  /* tensor */
  if(tensor->stride[0] == 1)
  {
    transpose = 'n';
    tensor_ = tensor;
  }
  else if(tensor->stride[1] == 1)
  {
    THTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    THTensor_(transpose)(tensor, NULL, 0, 1);
    THTensor_(transpose)(m1, NULL, 0, 1);
    THTensor_(transpose)(m2, NULL, 0, 1);
    transpose = 't';
    tensor_ = tensor;
  }
  else
  {
    transpose = 'n';
    THTensor_(transpose)(tensor, NULL, 0, 1);
    tensor_ = THTensor_(newClone)(tensor);
    THTensor_(transpose)(tensor, NULL, 0, 1);
    THTensor_(transpose)(tensor_, NULL, 0, 1);
  }

  /* m1 */
  if(m1->stride[0] == 1)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride[1] == 1)
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = 't';
    m1_ = THTensor_(newContiguous)(m1);
  }

  /* m2 */
  if(m2->stride[0] == 1)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride[1] == 1)
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = 't';
    m2_ = THTensor_(newContiguous)(m2);
  }

  /* do the operation */
  THBlas_(gemm)(transpose_m1,
                transpose_m2,
                tensor_->size[0],
                tensor_->size[1],
                m1_->size[1],
                alpha,
                THTensor_(data)(m1_),
                (transpose_m1 == 'n' ? m1_->stride[1] : m1_->stride[0]),
                THTensor_(data)(m2_),
                (transpose_m2 == 'n' ? m2_->stride[1] : m2_->stride[0]),
                beta,
                THTensor_(data)(tensor_),
                tensor_->stride[1]);

  /* free intermediate variables */
  if(m1_ != m1)
    THTensor_(free)(m1_);

  if(m2_ != m2)
    THTensor_(free)(m2_);

  if(tensor_ != tensor)
    THTensor_(freeCopyTo)(tensor_, tensor);

  if(transpose == 't')
  {
    THTensor_(transpose)(tensor, NULL, 0, 1);
    THTensor_(transpose)(m1, NULL, 0, 1);
    THTensor_(transpose)(m2, NULL, 0, 1);
  }
} 

#endif
