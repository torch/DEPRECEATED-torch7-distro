#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLab.c"
#else

void THLab_(add)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data + value;);
}

void THLab_(mul)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data * value;);
}

void THLab_(div)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data / value;);
}

void THLab_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data + value * *src_data;);
}

void THLab_(cmul)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data * *src_data;);
}

void THLab_(cdiv)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data / *src_data;);
}

void THLab_(numel)(long *n_, THTensor *t)
{
  *n_ = THTensor_(nElement)(t);
}

void THLab_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension)
{
  THLongStorage *dim;
  long i;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension out of range");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(values_, dim, NULL);
  THLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY3(real, t, real, values_, long, indices_, dimension,
                       long theIndex = 0;
                       real theMax = t_data[0];
                       for(i = 1; i < t_size; i++)
                       {
                         if(t_data[i*t_stride] > theMax)
                         {
                           theIndex = i;
                           theMax = t_data[i*t_stride];
                         }
                       }
                       *indices__data = theIndex;
                       *values__data = theMax;);  

}

void THLab_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension)
{
  THLongStorage *dim;
  long i;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension out of range");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(values_, dim, NULL);
  THLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY3(real, t, real, values_, long, indices_, dimension,
                       long theIndex = 0;
                       real theMin = t_data[0];
                       for(i = 1; i < t_size; i++)
                       {
                         if(t_data[i*t_stride] < theMin)
                         {
                           theIndex = i;
                           theMin = t_data[i*t_stride];
                         }
                       }
                       *indices__data = theIndex;
                       *values__data = theMin;);  

}


void THLab_(sum)(THTensor *r_, THTensor *t, int dimension)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension out of range");
  
  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                           real sum = 0;
                           long i;
                           for(i = 0; i < t_size; i++)
                             sum += t_data[i*t_stride];
                           *r__data = sum;);
}

void THLab_(prod)(THTensor *r_, THTensor *t, int dimension)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension out of range");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);
  
  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                           real prod = 1;
                           long i;
                           for(i = 0; i < t_size; i++)
                             prod *= t_data[i*t_stride];
                           *r__data = prod;);

}

void THLab_(cumsum)(THTensor *r_, THTensor *t, int dimension)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension out of range");
  
  THTensor_(resizeAs)(r_, t);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                           real cumsum = 0;
                           long i;
                           for(i = 0; i < t_size; i++)
                           {
                             cumsum += t_data[i*t_stride];
                             r__data[i*r__stride] = cumsum;
                           });
}

void THLab_(cumprod)(THTensor *r_, THTensor *t, int dimension)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension out of range");
  
  THTensor_(resizeAs)(r_, t);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                           real cumprod = 1;
                           long i;
                           for(i = 0; i < t_size; i++)
                           {
                             cumprod *= t_data[i*t_stride];
                             r__data[i*r__stride] = cumprod;
                           });
}

void THLab_(trace)(real *trace_, THTensor *t)
{
  real *t_data = THTensor_(data)(t);
  real sum = 0;
  long i = 0;
  long t_stride_0, t_stride_1, t_diag_size;

  THArgCheck(THTensor_(nDimension)(t) == 2, 1, "not a matrix");

  t_stride_0 = THTensor_(stride)(t, 0);
  t_stride_1 = THTensor_(stride)(t, 1);
  t_diag_size = THMin(THTensor_(size)(t, 0), THTensor_(size)(t, 1));
  while(i < t_diag_size)
  {
    sum += t_data[i*(t_stride_0+t_stride_1)];
    i++;
  }

  *trace_ = sum;
}

void THLab_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension)
{
  int i;

  if(THTensor_(nDimension)(a) != THTensor_(nDimension)(b))
    THError("inconsitent tensor sizes");
  
  for(i = 0; i < THTensor_(nDimension)(a); i++)
  {
    if(THTensor_(size)(a, i) != THTensor_(size)(b, i))
      THError("inconsistent tensor sizes");
  }
  
  if(dimension < 0)
  {
    for(i = 0; i < THTensor_(nDimension)(a); i++)
    {
      if(THTensor_(size)(a, i) == 3)
      {
        dimension = i;
        break;
      }
    }
    if(dimension < 0)
      THError("no dimension of size 3");
  }

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(a), 3, "dimension out of range");
  THArgCheck(THTensor_(size)(a, dimension) == 3, 3, "dimension size is not 3");

  THTensor_(resizeAs)(r_, a);

  TH_TENSOR_DIM_APPLY3(real, a, real, b, real, r_, dimension,
                           r__data[0*r__stride] = a_data[1*a_stride]*b_data[2*b_stride] - a_data[2*a_stride]*b_data[1*b_stride];
                           r__data[1*r__stride] = a_data[2*a_stride]*b_data[0*b_stride] - a_data[0*a_stride]*b_data[2*b_stride];
                           r__data[2*r__stride] = a_data[0*a_stride]*b_data[1*b_stride] - a_data[1*a_stride]*b_data[0*b_stride];);
}

void THLab_(zeros)(THTensor *r_, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(zero)(r_);
}

void THLab_(ones)(THTensor *r_, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(fill)(r_, 1);
}

void THLab_(diag)(THTensor *r_, THTensor *t, int k)
{
  THArgCheck(THTensor_(nDimension)(t) == 1 || THTensor_(nDimension)(t) == 2, 1, "matrix or a vector expected");

  if(THTensor_(nDimension)(t) == 1)
  {
    real *t_data = THTensor_(data)(t);
    long t_stride_0 = THTensor_(stride)(t, 0);
    long t_size = THTensor_(size)(t, 0);
    long sz = t_size + (k >= 0 ? k : -k);
    real *r__data;
    long r__stride_0;
    long r__stride_1;
    long i;

    THTensor_(resize2d)(r_, sz, sz);    
    THTensor_(zero)(r_);
    r__data = THTensor_(data)(r_);
    r__stride_0 = THTensor_(stride)(r_, 0);
    r__stride_1 = THTensor_(stride)(r_, 1);
    r__data += (k >= 0 ? k*r__stride_1 : -k*r__stride_0);

    for(i = 0; i < t_size; i++)
      r__data[i*(r__stride_0+r__stride_1)] = t_data[i*t_stride_0];
  }
  else
  {
    real *t_data = THTensor_(data)(t);
    long t_stride_0 = THTensor_(stride)(t, 0);
    long t_stride_1 = THTensor_(stride)(t, 1);
    long sz;
    real *r__data;
    long r__stride_0;
    long i;

    if(k >= 0)
      sz = THMin(THTensor_(size)(t, 0), THTensor_(size)(t, 1)-k);
    else
      sz = THMin(THTensor_(size)(t, 0)+k, THTensor_(size)(t, 1));
    THTensor_(resize1d)(r_, sz);
    r__data = THTensor_(data)(r_);
    r__stride_0 = THTensor_(stride)(r_, 0);

    t_data += (k >= 0 ? k*t_stride_1 : -k*t_stride_0);
    for(i = 0; i < sz; i++)
      r__data[i*r__stride_0] = t_data[i*(t_stride_0+t_stride_1)];
  }
}

void THLab_(eye)(THTensor *r_, long n, long m)
{
  real *r__data;
  long i, sz;

  THArgCheck(n > 0, 1, "invalid argument");

  if(m <= 0)
    m = n;

  THTensor_(resize2d)(r_, n, m);
  THTensor_(zero)(r_);

  i = 0;
  r__data = THTensor_(data)(r_);
  sz = THMin(THTensor_(size)(r_, 0), THTensor_(size)(r_, 1));
  for(i = 0; i < sz; i++)
    r__data[i*(r_->stride[0]+r_->stride[1])] = 1;
}


void THLab_(range)(THTensor *r_, real xmin, real xmax, real step)
{
  long size;
  real i = 0;

  THArgCheck(step > 0, 3, "step must be a positive number");
  THArgCheck(xmax > xmin, 2, "upper bound must be larger than lower bound");

  size = (long)((xmax-xmin)/step+1);
  
  THTensor_(resize1d)(r_, size);

  TH_TENSOR_APPLY(real, r_, *r__data = xmin + (i++)*step;);
}

void THLab_(randperm)(THTensor *r_, long n)
{
  real *r__data;
  long r__stride_0;
  long i;

  THArgCheck(n > 0, 1, "must be strictly positive");

  THTensor_(resize1d)(r_, n);
  r__data = THTensor_(data)(r_);
  r__stride_0 = THTensor_(stride)(r_,0);

  for(i = 0; i < n; i++)
    r__data[i*r__stride_0] = (real)(i);

  for(i = 0; i < n-1; i++)
  {    
    long z = THRandom_random() % (n-i);
    real sav = r__data[i*r__stride_0];
    r__data[i*r__stride_0] = r__data[(z+i)*r__stride_0];
    r__data[(z+i)*r__stride_0] = sav;
  }
}

void THLab_(reshape)(THTensor *r_, THTensor *t, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(copy)(r_, t);
}

/* I cut and pasted (slightly adapted) the quicksort code from
   http://www.alienryderflex.com/quicksort/
   This public-domain C implementation by Darel Rex Finley.
   Thanks man :)
*/
#define  MAX_LEVELS  300
static void THLab_(quicksortascend)(real *arr, long *idx, long elements, long stride)
{
  long beg[MAX_LEVELS], end[MAX_LEVELS], i=0, L, R, swap, pid;
  real piv;
  
  beg[0]=0; end[0]=elements;
  while (i>=0) {
    L=beg[i]; R=end[i]-1;
    if (L<R) {
      piv=arr[L*stride];
      pid=idx[L*stride];
      while (L<R) {
        while (arr[R*stride]>=piv && L<R) R--; if (L<R) {idx[L*stride]=idx[R*stride]; arr[L*stride]=arr[R*stride]; L++;}
        while (arr[L*stride]<=piv && L<R) L++; if (L<R) {idx[R*stride]=idx[L*stride]; arr[R*stride]=arr[L*stride]; R--;} }
      idx[L*stride]=pid; arr[L*stride]=piv; beg[i+1]=L+1; end[i+1]=end[i]; end[i++]=L;
      if (end[i]-beg[i]>end[i-1]-beg[i-1]) {
        swap=beg[i]; beg[i]=beg[i-1]; beg[i-1]=swap;
        swap=end[i]; end[i]=end[i-1]; end[i-1]=swap; }}
    else {
      i--; }}}

static void THLab_(quicksortdescend)(real *arr, long *idx, long elements, long stride)
{
  long beg[MAX_LEVELS], end[MAX_LEVELS], i=0, L, R, swap, pid;
  real piv;
  
  beg[0]=0; end[0]=elements;
  while (i>=0) {
    L=beg[i]; R=end[i]-1;
    if (L<R) {
      piv=arr[L*stride];
      pid=idx[L*stride];
      while (L<R) {
        while (arr[R*stride]<=piv && L<R) R--; if (L<R) {idx[L*stride]=idx[R*stride]; arr[L*stride]=arr[R*stride]; L++;}
        while (arr[L*stride]>=piv && L<R) L++; if (L<R) {idx[R*stride]=idx[L*stride]; arr[R*stride]=arr[L*stride]; R--;} }
      idx[L*stride]=pid; arr[L*stride]=piv; beg[i+1]=L+1; end[i+1]=end[i]; end[i++]=L;
      if (end[i]-beg[i]>end[i-1]-beg[i-1]) {
        swap=beg[i]; beg[i]=beg[i-1]; beg[i-1]=swap;
        swap=end[i]; end[i]=end[i-1]; end[i-1]=swap; }}
    else {
      i--; }}}

void THLab_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder)
{
  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "invalid dimension");

  THTensor_(resizeAs)(rt_, t);
  THTensor_(copy)(rt_, t);

  {
    THLongStorage *size = THTensor_(newSizeOf)(t);
    THLongTensor_resize(ri_, size, NULL);
    THLongStorage_free(size);
  }

  if(descendingOrder)
  {
    TH_TENSOR_DIM_APPLY2(real, rt_, long, ri_, dimension, 
                         long i;
                         for(i = 0; i < ri__size; i++)
                           ri__data[i*ri__stride] = i;
                         THLab_(quicksortdescend)(rt__data, ri__data, rt__size, rt__stride);)
  }
  else
  {
    TH_TENSOR_DIM_APPLY2(real, rt_, long, ri_, dimension,
                         long i;
                         for(i = 0; i < ri__size; i++)
                           ri__data[i*ri__stride] = i;
                         THLab_(quicksortascend)(rt__data, ri__data, rt__size, rt__stride);)
  }
}

void THLab_(tril)(THTensor *r_, THTensor *t, long k)
{
  long t_size_0, t_size_1;
  long t_stride_0, t_stride_1;
  long r__stride_0, r__stride_1;
  real *t_data, *r__data;
  long r, c;

  THArgCheck(THTensor_(nDimension)(t) == 2, 1, "not a matrix");

  THTensor_(resizeAs)(r_, t);

  t_size_0 = THTensor_(size)(t, 0);
  t_size_1 = THTensor_(size)(t, 1);
  t_stride_0 = THTensor_(stride)(t, 0);
  t_stride_1 = THTensor_(stride)(t, 1);
  r__stride_0 = THTensor_(stride)(r_, 0);
  r__stride_1 = THTensor_(stride)(r_, 1);
  r__data = THTensor_(data)(r_);
  t_data = THTensor_(data)(t);

  for(r = 0; r < t_size_0; r++)
  {
    long sz = THMin(r+k+1, t_size_1);
    for(c = THMax(0, r+k); c < t_size_1; c++)
      r__data[r*r__stride_0+c*r__stride_1] = 0;
    for(c = 0; c < sz; c++)
      r__data[r*r__stride_0+c*r__stride_1] = t_data[r*t_stride_0+c*t_stride_1];
  }
}

void THLab_(triu)(THTensor *r_, THTensor *t, long k)
{
  long t_size_0, t_size_1;
  long t_stride_0, t_stride_1;
  long r__stride_0, r__stride_1;
  real *t_data, *r__data;
  long r, c;

  THArgCheck(THTensor_(nDimension)(t) == 2, 1, "not a matrix");

  THTensor_(resizeAs)(r_, t);

  t_size_0 = THTensor_(size)(t, 0);
  t_size_1 = THTensor_(size)(t, 1);
  t_stride_0 = THTensor_(stride)(t, 0);
  t_stride_1 = THTensor_(stride)(t, 1);
  r__stride_0 = THTensor_(stride)(r_, 0);
  r__stride_1 = THTensor_(stride)(r_, 1);
  r__data = THTensor_(data)(r_);
  t_data = THTensor_(data)(t);

  for(r = 0; r < t_size_0; r++)
  {
    long sz = THMin(r+k, t_size_1);
    for(c = THMax(0, r+k); c < t_size_1; c++)
      r__data[r*r__stride_0+c*r__stride_1] = t_data[r*t_stride_0+c*t_stride_1];
    for(c = 0; c < sz; c++)
      r__data[r*r__stride_0+c*r__stride_1] = 0;
  }
}

void THLab_(cat)(THTensor *r_, THTensor *ta, THTensor *tb, int dimension)
{
  THLongStorage *size;
  int i;
  int ndim = THMax(ta->nDimension, tb->nDimension);
  ndim = THMax(ndim, dimension+1);

  THArgCheck(dimension >= 0, 4, "invalid dimension");

  size = THLongStorage_newWithSize(ndim);
  for(i = 0; i < ndim; i++)
  {
    int tadi = (i < ta->nDimension ? ta->size[i] : 1);
    int tbdi = (i < tb->nDimension ? tb->size[i] : 1);

    if(i == dimension)
      size->data[i] = tadi+tbdi;
    else
    {
      if(tadi != tbdi)
      {
        THLongStorage_free(size);
        THError("inconsistent tensor sizes");
      }
      size->data[i] = tadi;
    }
  }

  THTensor_(resize)(r_, size, NULL);
  THLongStorage_free(size);

  {
    THTensor *nta = THTensor_(newWithTensor)(r_);
    THTensor_(narrow)(nta, NULL, dimension, 0, (dimension < ta->nDimension ? ta->size[dimension] : 1));
    THTensor_(copy)(nta, ta);
    THTensor_(free)(nta);
  }

  {
    THTensor *ntb = THTensor_(newWithTensor)(r_);
    THTensor_(narrow)(ntb, NULL, dimension, (dimension < ta->nDimension ? ta->size[dimension] : 1), (dimension < tb->nDimension ? tb->size[dimension] : 1));
    THTensor_(copy)(ntb, tb);
    THTensor_(free)(ntb);
  }
}

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
void THLab_(conv2DRevger)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol)
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
  
  long k,i;
  for(k = 0; k < nKernelPlane; k++)
  {
    /* get kernel */
    real *ptr_weight = weight_data+k*kstride0;

    for(i = 0; i < nInputPlane; i++)
    {
      /* get input */
      real *ptr_input = input_data+i*istride0;

      /* do image, kernel convolution */
      THLab_(validConv2DRevptr)(output_data,
				ptr_input,  nInputRows,  nInputCols,
				ptr_weight, nKernelRows, nKernelCols,
				srow, scol);
      /* Next output plane */
      output_data += nOutputCols*nOutputRows;
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
void THLab_(conv2Dger)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char *type)
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
  
  long k,i;
  for(k = 0; k < nKernelPlane; k++)
  {
    /* get kernel */
    real *ptr_weight = weight_data+k*kstride0;

    for(i = 0; i < nInputPlane; i++)
    {
      /* get input */
      real *ptr_input = input_data+i*istride0;

      /* do image, kernel convolution */
      if (*type == 'f')
	THLab_(fullConv2Dptr)(output_data,
			      ptr_input,  nInputRows,  nInputCols,
			      ptr_weight, nKernelRows, nKernelCols,
			      srow, scol);
      else
	THLab_(validConv2Dptr)(output_data,
			       ptr_input,  nInputRows,  nInputCols,
			       ptr_weight, nKernelRows, nKernelCols,
			       srow, scol);
      /* Next output plane */
      output_data += nOutputCols*nOutputRows;
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
void THLab_(conv2Dmv)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char *type)
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
  
  long k,i;
  for(k = 0; k < nOutputPlane; k++)
  {
    for(i = 0; i < nInputPlane; i++)
    {
      /* get kernel */
      real *ptr_weight = weight_data + k*kstride0 + i*kstride1;
      /* get input */
      real *ptr_input = input_data + i*istride0;

      /* do image, kernel convolution */
      if (*type == 'f')
	THLab_(fullConv2Dptr)(output_data,
			      ptr_input,  nInputRows,  nInputCols,
			      ptr_weight, nKernelRows, nKernelCols,
			      srow, scol);
      else
	THLab_(validConv2Dptr)(output_data,
			       ptr_input,  nInputRows,  nInputCols,
			       ptr_weight, nKernelRows, nKernelCols,
			       srow, scol);
    }
    /* Next output plane */
    output_data += nOutputCols*nOutputRows;
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}

/* 
   2D input, 2D kernel, 2D output
   scalar multiplication like
   y <- x*y + beta*y
 */
void THLab_(conv2Dmul)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char *type)
{
					 
  THArgCheck(t_->nDimension == 2 , 3, "input: 2D Tensor expected");
  THArgCheck(k_->nDimension == 2 , 4, "kernel: 2D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(*type == 'v' || *type == 'f', 7, "type of convolution can 'v' or 'f'");

  THTensor *input = THTensor_(newContiguous)(t_, 0);
  THTensor* kernel = THTensor_(newContiguous)(k_, 0);

  long nInputRows  = input->size[0];
  long nInputCols  = input->size[1];
  long nKernelRows = kernel->size[0];
  long nKernelCols = kernel->size[1];
  long nOutputRows, nOutputCols;

  THArgCheck(nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2, "Input image is smaller than kernel");
  
  if (*type == 'f') {
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;
  } else { // valid
    nOutputRows = (nInputRows - nKernelRows) / srow + 1;
    nOutputCols = (nInputCols - nKernelCols) / scol + 1;
  }

  long nelem = THTensor_(nElement)(r_);
  THTensor_(resize2d)(r_, nOutputRows, nOutputCols);
  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
    THTensor_(zero)(r_);
  else if (beta != 1)
    THTensor_(mul)(r_, beta);

  real *ptr_input = THTensor_(data)(input);
  real *ptr_weight = THTensor_(data)(kernel);
  real *output_data = THTensor_(data)(r_);  
  
  
  /* do image, kernel convolution */
  if (*type == 'f')
    THLab_(fullConv2Dptr)(output_data,
			  ptr_input,  nInputRows,  nInputCols,
			  ptr_weight, nKernelRows, nKernelCols,
			  srow, scol);
  else
    THLab_(validConv2Dptr)(output_data,
			   ptr_input,  nInputRows,  nInputCols,
			   ptr_weight, nKernelRows, nKernelCols,
			   srow, scol);
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}

/* floating point only now */

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

#define LAB_IMPLEMENT_BASIC_FUNCTION(NAME, CFUNC)             \
  void THLab_(NAME)(THTensor *r_, THTensor *t)                \
  {                                                           \
    THTensor_(resizeAs)(r_, t);                               \
    TH_TENSOR_APPLY2(real, t, real, r_, *r__data = CFUNC(*t_data);); \
  }                                                           \

#define LAB_IMPLEMENT_BASIC_FUNCTION_VALUE(NAME, CFUNC)              \
  void THLab_(NAME)(THTensor *r_, THTensor *t, real value)           \
  {                                                                  \
    THTensor_(resizeAs)(r_, t);                                      \
    TH_TENSOR_APPLY2(real, t, real, r_, *r__data = CFUNC(*t_data, value);); \
  }                                                                  \

LAB_IMPLEMENT_BASIC_FUNCTION(log,log)
LAB_IMPLEMENT_BASIC_FUNCTION(log1p,log1p)
LAB_IMPLEMENT_BASIC_FUNCTION(exp,exp)
LAB_IMPLEMENT_BASIC_FUNCTION(cos,cos)
LAB_IMPLEMENT_BASIC_FUNCTION(acos,acos)
LAB_IMPLEMENT_BASIC_FUNCTION(cosh,cosh)
LAB_IMPLEMENT_BASIC_FUNCTION(sin,sin)
LAB_IMPLEMENT_BASIC_FUNCTION(asin,asin)
LAB_IMPLEMENT_BASIC_FUNCTION(sinh,sinh)
LAB_IMPLEMENT_BASIC_FUNCTION(tan,tan)
LAB_IMPLEMENT_BASIC_FUNCTION(atan,atan)
LAB_IMPLEMENT_BASIC_FUNCTION(tanh,tanh)
LAB_IMPLEMENT_BASIC_FUNCTION_VALUE(pow,pow)
LAB_IMPLEMENT_BASIC_FUNCTION(sqrt,sqrt)
LAB_IMPLEMENT_BASIC_FUNCTION(ceil,ceil)
LAB_IMPLEMENT_BASIC_FUNCTION(floor,floor)
LAB_IMPLEMENT_BASIC_FUNCTION(abs,fabs)

void THLab_(mean)(THTensor *r_, THTensor *t, int dimension)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "invalid dimension");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                           real sum = 0;
                           long i;
                           for(i = 0; i < t_size; i++)
                             sum += t_data[i*t_stride];
                           *r__data = sum/t_size;);
}

void THLab_(std)(THTensor *r_, THTensor *t, int dimension, int flag)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 3, "invalid dimension");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                           real sum = 0;
                           real sum2 = 0;
                           long i;
                           for(i = 0; i < t_size; i++)
                           {
                             real z = t_data[i*t_stride];
                             sum += z;
                             sum2 += z*z;
                           }

                           if(flag)
                           {
                             sum /= t_size;
                             sum2 /= t_size;
                             sum2 -= sum*sum;
                             sum2 = (sum2 < 0 ? 0 : sum2);
                             *r__data = sqrt(sum2);
                           }
                           else
                           {
                             sum /= t_size;
                             sum2 /= t_size-1;
                             sum2 -= ((real)t_size)/((real)(t_size-1))*sum*sum;
                             sum2 = (sum2 < 0 ? 0 : sum2);
                             *r__data = sqrt(sum2);
                           });
}

void THLab_(var)(THTensor *r_, THTensor *t, int dimension, int flag)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 3, "invalid dimension");

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                           real sum = 0;
                           real sum2 = 0;
                           long i;
                           for(i = 0; i < t_size; i++)
                           {
                             real z = t_data[i*t_stride];
                             sum += z;
                             sum2 += z*z;
                           }

                           if(flag)
                           {
                             sum /= t_size;
                             sum2 /= t_size;
                             sum2 -= sum*sum;
                             sum2 = (sum2 < 0 ? 0 : sum2);
                             *r__data = sum2;
                           }
                           else
                           {
                             sum /= t_size;
                             sum2 /= t_size-1;
                             sum2 -= ((real)t_size)/((real)(t_size-1))*sum*sum;
                             sum2 = (sum2 < 0 ? 0 : sum2);
                             *r__data = sum2;
                           });
}

void THLab_(norm)(real *norm_, THTensor *t, real value)
{
  *norm_ = THTensor_(norm)(t, value);
}

void THLab_(dist)(real *dist_, THTensor *a, THTensor *b, real value)
{ 
  *dist_ = THTensor_(dist)(a, b, value);
}

void THLab_(linspace)(THTensor *r_, real a, real b, long n)
{
  real i = 0;

  THArgCheck(n > 0, 3, "invalid number of points");
  THArgCheck(a <= b, 2, "end range should be greater than start range");
  
  THTensor_(resize1d)(r_, n);

  TH_TENSOR_APPLY(real, r_,
                  *r__data = a + i*(b-a)/((real)(n-1));
                  i++;
    );
}

void THLab_(logspace)(THTensor *r_, real a, real b, long n)
{
  real i = 0;

  THArgCheck(n > 0, 3, "invalid number of points");
  THArgCheck(a <= b, 2, "end range should be greater than start range");
  
  THTensor_(resize1d)(r_, n);

  TH_TENSOR_APPLY(real, r_,
                  *r__data = pow(10.0, a + i*(b-a)/((real)(n-1)));
                  i++;
    );
}

void THLab_(rand)(THTensor *r_, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(uniform)(r_, 0, 1);
}

void THLab_(randn)(THTensor *r_, THLongStorage *size)
{
  THTensor_(resize)(r_, size, NULL);
  THTensor_(normal)(r_, 0, 1);
}

#endif /* floating point only part */
#endif
