#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THOmpLab.c"
#else

/*
  2D Input, 2D kernel  : convolve given image with the given kernel.
*/
static void THLab_(validConv2Dptr)(real *r_,
				   real *t_, long ir, long ic, 
				   real *k_, long kr, long kc, 
				   long sr, long sc)
{
  long or = (ir - kr) / sr + 1;
  long oc = (ic - kc) / sc + 1;

  long xx, yy;  

  for(yy = 0; yy < or; yy++)
  {
    for(xx = 0; xx < oc; xx++)
    {
      /* Dot product in two dimensions... (between input image and the mask) */
      real *pi_ = t_ + yy*sr*ic + xx*sc;
      real *pw_ = k_;
      real sum = 0;
      long kx, ky;
      for(ky = 0; ky < kr; ky++)
      {
	for(kx = 0; kx < kc; kx++) {
	  sum += pi_[kx]*pw_[kx];
	}
	pi_ += ic; /* next input line */
	pw_ += kc; /* next mask line */
      }
      /* Update output */
      *r_ += sum;
      *r_++;
    }
  }
}

/*
  2D Input, 2D kernel  : convolve given image with the given kernel, full convolution.
*/
static void THLab_(fullConv2Dptr)(real *r_,
				  real *t_, long ir, long ic, 
				  real *k_, long kr, long kc, 
				  long sr, long sc)
{
  long or = (ir - 1) * sr + kr;
  long oc = (ic - 1) * sc + kc;

  long xx, yy;

  for(yy = 0; yy < ir; yy++)
  {
    for(xx = 0; xx < ic; xx++)
    {
      /* Outer product in two dimensions... (between input image and the mask) */
      real *po_ = r_ + yy*sr*oc + xx*sc;
      real *pw_ = k_;
      real sum = 0;
      long kx, ky;
      for(ky = 0; ky < kr; ky++)
      {
	for(kx = 0; kx < kc; kx++) {
	  po_[kx] += *t_ * pw_[kx];
	}
	po_ += oc; /* next input line */
	pw_ += kc; /* next mask line */
      }
      t_++;
    }
  }
}

/*
  2D Input, 2D kernel  : convolve given image with the given kernel, valid convolution.
  for sr,sc=1 this is equivalent to validConv2Dptr, but otherwise it is useful for
  calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
*/
static void THLab_(validConv2DRevptr)(real *r_,
				      real *t_, long ir, long ic, 
				      real *k_, long kr, long kc, 
				      long sr, long sc)
{
  long or = ir - (kr - 1) * sr;
  long oc = ic - (kc - 1) * sc;

  long xx, yy;
  for(yy = 0; yy < kr; yy++)
  {
    for(xx = 0; xx < kc; xx++)
    {
      real *po_ = r_;
      real *pi_ = t_ + yy*sr*ic + xx*sc;
      real z = *k_++;
      long kx, ky;
      
      for(ky = 0; ky < or; ky++)
      {
	for(kx = 0; kx < oc; kx++)
	  po_[kx] += z * pi_[kx];
	pi_ += ic;
	po_ += oc;
      }
    }
  }
}


/* 
   3D input, 3D kernel, 4D output
   like rank1 update
   A <- xx' + beta*A
   for sr,sc=1 this is equivalent to conv2Dger, but otherwise it is useful for
   calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
 */
void THOmpLab_(conv2DRevger)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelPlane, nKernelRows, nKernelCols;
  long nOutputPlane, nOutputRows, nOutputCols;
  long istride0, kstride0;
  
  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");

  THTensor *input = THTensor_(newContiguous)(t_, 0);
  THTensor *kernel = THTensor_(newContiguous)(k_, 0);

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];
  
  kstride0 = kernel->stride[0];
  nKernelPlane = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];
  nOutputPlane = nInputPlane * kernel->size[0];

  THArgCheck(nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2, "covn2DRevger : Input image is smaller than kernel");

  nOutputRows = nInputRows - (nKernelRows - 1) * srow;
  nOutputCols = nInputCols - (nKernelCols - 1) * scol;

  long nelem = THTensor_(nElement)(r_);
  THTensor_(resize4d)(r_,nKernelPlane, nInputPlane, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    THTensor_(zero)(r_);
  }
  else if (beta != 1)
    THTensor_(mul)(r_, beta);

  real *input_data = THTensor_(data)(input);
  real *weight_data = THTensor_(data)(kernel);
  real *output_data = THTensor_(data)(r_);  
  
  long k;
#pragma omp parallel for 
  for(k = 0; k < nKernelPlane; k++)
  {
    long i;
    /* get kernel */
    real *ptr_weight = weight_data+k*kstride0;

    for(i = 0; i < nInputPlane; i++)
    {
      /* get output */
      real *ptr_output = output_data + k*nInputPlane*nOutputCols*nOutputRows + i*nOutputCols*nOutputRows;
      /* get input */
      real *ptr_input = input_data+i*istride0;

      /* do image, kernel convolution */
      THLab_(validConv2DRevptr)(ptr_output,
				ptr_input,  nInputRows,  nInputCols,
				ptr_weight, nKernelRows, nKernelCols,
				srow, scol);
      /* Next output plane */
      /* output_data += nOutputCols*nOutputRows; */
    }
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}

      
/* 
   3D input, 3D kernel, 4D output
   like rank1 update
   A <- xx' + beta*A
 */
void THOmpLab_(conv2Dger)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char *type)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelPlane, nKernelRows, nKernelCols;
  long nOutputPlane, nOutputRows, nOutputCols;
  long istride0, kstride0;
					 
  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(*type == 'v' || *type == 'f', 7, "type of convolution can 'v' or 'f'");

  THTensor *input = THTensor_(newContiguous)(t_, 0);
  THTensor *kernel = THTensor_(newContiguous)(k_, 0);

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];
  
  kstride0 = kernel->stride[0];
  nKernelPlane = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];
  nOutputPlane = nInputPlane * kernel->size[0];

  THArgCheck(nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2, "conv2Dger : Input image is smaller than kernel");

  if (*type == 'f') {
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;
  } else { // valid
    nOutputRows = (nInputRows - nKernelRows) / srow + 1;
    nOutputCols = (nInputCols - nKernelCols) / scol + 1;
  }

  long nelem = THTensor_(nElement)(r_);
  THTensor_(resize4d)(r_,nKernelPlane, nInputPlane, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    THTensor_(zero)(r_);
  }
  else if (beta != 1)
    THTensor_(mul)(r_, beta);

  real *input_data = THTensor_(data)(input);
  real *weight_data = THTensor_(data)(kernel);
  real *output_data = THTensor_(data)(r_);  
  
  long k;
#pragma omp parallel for
  for(k = 0; k < nKernelPlane; k++)
  {
    long i;
    /* get kernel */
    real *ptr_weight = weight_data+k*kstride0;

    for(i = 0; i < nInputPlane; i++)
    {
      /* get output */
      real *ptr_output = output_data + k*nInputPlane*nOutputCols*nOutputRows + i*nOutputCols*nOutputRows;
      /* get input */
      real *ptr_input = input_data+i*istride0;

      /* do image, kernel convolution */
      if (*type == 'f')
	THLab_(fullConv2Dptr)(ptr_output,
			      ptr_input,  nInputRows,  nInputCols,
			      ptr_weight, nKernelRows, nKernelCols,
			      srow, scol);
      else
	THLab_(validConv2Dptr)(ptr_output,
			       ptr_input,  nInputRows,  nInputCols,
			       ptr_weight, nKernelRows, nKernelCols,
			       srow, scol);
      /* Next output plane */
      /* output_data += nOutputCols*nOutputRows; */
    }
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}

/* 
   3D input, 4D kernel, 3D output
   matrix vector product like
   y <- Ax + beta*y
 */
void THOmpLab_(conv2Dmv)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char *type)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelRows, nKernelCols;
  long nOutputPlane, nOutputRows, nOutputCols;
  long istride0, kstride0, kstride1;
					 
  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(*type == 'v' || *type == 'f', 7, "type of convolution can 'v' or 'f'");

  THTensor *input = THTensor_(newContiguous)(t_, 0);
  THTensor* kernel;
  if (!(k_->stride[3] == 1) || !(k_->stride[2] == k_->size[3])) {
    kernel = THTensor_(newContiguous)(k_, 0);
  } else {
    THTensor_(retain)(k_);
    kernel = k_;
  }

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];
  
  kstride0    = kernel->stride[0];
  kstride1    = kernel->stride[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];
  nOutputPlane = kernel->size[0];
  THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");

  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *type == 'f', 2, "conv2Dmv : Input image is smaller than kernel");
  
  if (*type == 'f') {
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;
  } else { // valid
    nOutputRows = (nInputRows - nKernelRows) / srow + 1;
    nOutputCols = (nInputCols - nKernelCols) / scol + 1;
  }

  long nelem = THTensor_(nElement)(r_);
  THTensor_(resize3d)(r_, nOutputPlane, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    THTensor_(zero)(r_);
  }
  else if (beta != 1)
    THTensor_(mul)(r_, beta);

  real *input_data = THTensor_(data)(input);
  real *weight_data = THTensor_(data)(kernel);
  real *output_data = THTensor_(data)(r_);  
  
  long k;
#pragma omp parallel for
  for(k = 0; k < nOutputPlane; k++)
  {
    long i;
    /* get output */
    real *ptr_output = output_data + k*nOutputCols*nOutputRows;
    for(i = 0; i < nInputPlane; i++)
    {
      /* get kernel */
      real *ptr_weight = weight_data + k*kstride0 + i*kstride1;
      /* get input */
      real *ptr_input = input_data + i*istride0;

      /* do image, kernel convolution */
      if (*type == 'f')
	THLab_(fullConv2Dptr)(ptr_output,
			      ptr_input,  nInputRows,  nInputCols,
			      ptr_weight, nKernelRows, nKernelCols,
			      srow, scol);
      else
	THLab_(validConv2Dptr)(ptr_output,
			       ptr_input,  nInputRows,  nInputCols,
			       ptr_weight, nKernelRows, nKernelCols,
			       srow, scol);
    }
    /* Next output plane */
    /* output_data += nOutputCols*nOutputRows;*/
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}

#endif
