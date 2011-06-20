#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMath.c"
#else

void THTensor_(fill)(THTensor *tensor, real value)
{
  TH_TENSOR_APPLY(real, tensor, *tensor_data = value;);
}

void THTensor_(zero)(THTensor *tensor)
{
  TH_TENSOR_APPLY(real, tensor, *tensor_data = 0;);
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
  TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data *= *src_data;);
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

real THTensor_(dot)(THTensor *tensor, THTensor *src)
{
  real sum = 0;
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

real THTensor_(sum)(THTensor *tensor)
{
  real sum = 0;
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
real THTensor_(mean)(THTensor *tensor)
{ 
  THArgCheck(tensor->nDimension > 0, 1, "empty Tensor");
  return THTensor_(sum)(tensor)/THTensor_(nElement)(tensor);
}  

real THTensor_(var)(THTensor *tensor)
{ 
  real mean = THTensor_(mean)(tensor);
  real sum = 0;
  TH_TENSOR_APPLY(real, tensor, sum += (*tensor_data - mean)*(*tensor_data - mean););
  sum /= (THTensor_(nElement)(tensor)-1);
  return sum;
}

real THTensor_(std)(THTensor *tensor)
{ 
  return sqrt(THTensor_(var)(tensor));
} 
 
real THTensor_(norm)(THTensor *tensor, real value)
{ 
  real sum = 0;
  TH_TENSOR_APPLY(real, tensor, sum += pow(fabs(*tensor_data), value););
  return pow(sum, 1.0/value);
}

real THTensor_(dist)(THTensor *tensor, THTensor *src, real value)
{ 
  real sum = 0;
  TH_TENSOR_APPLY2(real, tensor, real, src, 
	sum += pow(fabs(*tensor_data - *src_data), value);)
  return pow(sum, 1.0/value);
}

void THTensor_(addmv)(THTensor *tensor, real alpha, THTensor *mat, THTensor *vec) 
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
                  1, THTensor_(data)(tensor), tensor->stride[0]);
  }
  else if(mat->stride[1] == 1)
  {
    THBlas_(gemv)('t',  mat->size[1], mat->size[0],
                  alpha, THTensor_(data)(mat), mat->stride[0],
                  THTensor_(data)(vec), vec->stride[0],
                  1, THTensor_(data)(tensor), tensor->stride[0]);
  }
  else
  {
    THTensor *cmat = THTensor_(newContiguous)(mat, 0);

    THBlas_(gemv)('t',  mat->size[1], mat->size[0],
                  alpha, THTensor_(data)(cmat), cmat->stride[0],
                  THTensor_(data)(vec), vec->stride[0],
                  1, THTensor_(data)(tensor), tensor->stride[0]);

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
    THTensor *ctensor = THTensor_(newContiguous)(tensor, 1);

    THBlas_(ger)(vec2->size[0], vec1->size[0],
                 alpha, THTensor_(data)(vec2), vec2->stride[0],
                 THTensor_(data)(vec1), vec1->stride[0],
                 THTensor_(data)(ctensor), ctensor->stride[0]);

    THTensor_(copy)(tensor, ctensor);
    THTensor_(free)(ctensor);
  }
}

void THTensor_(addmm)(THTensor *tensor, real alpha, THTensor *m1, THTensor *m2) 
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
    tensor_ = THTensor_(newContiguous)(tensor, 1);
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
    m1_ = THTensor_(newContiguous)(m1, 0);
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
    m2_ = THTensor_(newContiguous)(m2, 0);
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
                1,
                THTensor_(data)(tensor_),
                tensor_->stride[1]);

  /* free intermediate variables */
  if(m1_ != m1)
    THTensor_(free)(m1_);

  if(m2_ != m2)
    THTensor_(free)(m2_);

  if(tensor_ != tensor)
  {
    THTensor_(copy)(tensor, tensor_);
    THTensor_(free)(tensor_);
  }

  if(transpose == 't')
  {
    THTensor_(transpose)(tensor, NULL, 0, 1);
    THTensor_(transpose)(m1, NULL, 0, 1);
    THTensor_(transpose)(m2, NULL, 0, 1);
  }
} 

THTensor* THTensor_(newconv2_valid)(THTensor *image, THTensor *filter, long srow, long scol)
{
  THTensor *output = THTensor_(new)();
  THTensor_(conv2_valid)(output,image,filter,srow,scol);
  return output;
}

void THTensor_(conv2_valid)(THTensor *output, THTensor *image, THTensor *filter, long srow, long scol)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelRows, nKernelCols;
  long nOutputPlane, nOutputRows, nOutputCols;
  long istride0, kstride0, kstride1;
					 
  THArgCheck(image->nDimension == 3 || image->nDimension == 2 , 2, "2D or 3D Tensor expected");
  THArgCheck(filter->nDimension >=2 && filter->nDimension <= 4, 3, "2D or 3D or 4D Tensor expected");
  THArgCheck(srow >= 1, 4, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 4, "Stride should be a positive integer");

  THTensor *input = THTensor_(newContiguous)(image, 0);
  THTensor *kernel = THTensor_(newContiguous)(filter, 0);

  /*
    2D Input, 2D kernel  : convolve given image with the given kernel.
    2D Input, 3D kernels : convolve given image with all kernels.
    2D Input, 4D kernels : Only OK, if kernel->size[1] == 1
    3D Input, 2D kernel  : convolve each input image with the given kernel.
    3D Input, 3D kernels : convolve each given image with all kernels.
    3D Input, 4D kernels : regular full connected kernel assumed (out,in,kx,ky)
   */

  if (input->nDimension == 3) {
    nInputPlane = input->size[0];
    istride0    = input->stride[0];
    nInputRows  = input->size[1];
    nInputCols  = input->size[2];
  } else {
    nInputPlane = 1;
    istride0    = 0; 
    nInputRows  = input->size[0];
    nInputCols  = input->size[1];
  }
  
  if (kernel->nDimension == 2) {
    kstride0 = 0;
    kstride1 = 0;
    nKernelRows = kernel->size[0];
    nKernelCols = kernel->size[1];
    nOutputPlane = nInputPlane;
  } else if (kernel->nDimension == 3) {
    kstride0 = kernel->stride[0];
    kstride1 = 0;
    nKernelRows = kernel->size[1];
    nKernelCols = kernel->size[2];
    nOutputPlane = nInputPlane * kernel->size[0];
  } else {// if (kernel->nDimension == 4) {
    kstride0    = kernel->stride[0];
    kstride1    = kernel->stride[1];
    nKernelRows = kernel->size[2];
    nKernelCols = kernel->size[3];
    nOutputPlane = kernel->size[0];
    if (input->nDimension == 2)
      THArgCheck(kernel->size[1] == 1, 3, "2D input requires 4D kernels with size[1]=1");
  }

  THArgCheck(nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2, "Input image is smaller than kernel");

  nOutputRows = (nInputRows - nKernelRows) / srow + 1;
  nOutputCols = (nInputCols - nKernelCols) / scol + 1;

  if (input->nDimension == 2 && kernel->nDimension == 2)
    THTensor_(resize2d)(output, nOutputRows, nOutputCols);
  else
    THTensor_(resize3d)(output, nOutputPlane, nOutputRows, nOutputCols);

  if (input->nDimension == 3 && kernel->nDimension <= 3) {
    long nk = 1;
    if (kernel->nDimension == 3)
      nk = kernel->size[0];
    THTensor *outn = THTensor_(new)();
    THTensor *imn = THTensor_(new)();
    long i;
    for (i=0; i<nInputPlane; i++) {
      THTensor_(narrow)(outn,output,0,i*nk,nk);
      THTensor_(select)(imn,input,0,i);
      THTensor_(conv2_valid)(outn,imn,kernel,srow,scol);
    }
    THTensor_(free)(outn);
    THTensor_(free)(imn);
    return;
  }


  real *input_data = THTensor_(data)(input);
  real *weight_data = THTensor_(data)(kernel);
  real *output_data = THTensor_(data)(output);  
  
  long k,i;
  for(k = 0; k < nOutputPlane; k++)
  {
    // set all outputs to zero
    for(i = 0; i < nOutputCols*nOutputRows; i++)
      output_data[i] = 0;

    for(i = 0; i < nInputPlane; i++)
    {
      long xx, yy;

      /* Get the good mask for (k,i) (k out, i in) */
      real *ptr_weight = weight_data+k*kstride0+i*kstride1;
      
      /* Get the input image */
      real *ptr_input = input_data+i*istride0;
      
      /* For all output pixels... */
      real *ptr_output = output_data;
      for(yy = 0; yy < nOutputRows; yy++)
      {
        for(xx = 0; xx < nOutputCols; xx++)
        {
          /* Dot product in two dimensions... (between input image and the mask) */
          real *ptr_input_ = ptr_input+yy*srow*nInputCols+xx*scol;
          real *ptr_weight_ = ptr_weight;
          real sum = 0;
          long kx, ky;
          for(ky = 0; ky < nKernelRows; ky++)
          {
            for(kx = 0; kx < nKernelCols; kx++)
              sum += ptr_input_[kx]*ptr_weight_[kx];
            ptr_input_ += nInputCols; /* next input line */
            ptr_weight_ += nKernelCols; /* next mask line */
          }
          /* Update output */
          *ptr_output++ += sum;
        }
      }
    }

    /* Next output plane */
    output_data += nOutputCols*nOutputRows;
  }

  THTensor_(free)(input);
  THTensor_(free)(kernel);
}


#endif
