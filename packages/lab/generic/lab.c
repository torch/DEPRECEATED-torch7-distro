#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/lab.c"
#else

static int lab_(numel)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_(Tensor_id));
  long n;

  THLab_(numel)(&n, tensor);
  lua_pushnumber(L, n);

  return 1;
}

static int lab_(max_)(lua_State *L)
{
  THTensor *values_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THLongTensor *indices_ = luaT_checkudata(L, 2, torch_LongTensor_id);
  THTensor *t = luaT_checkudata(L, 3, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 4, THTensor_(nDimension)(t)))-1;

  THLab_(max)(values_, indices_, t, dimension);
  THLongTensor_add(indices_, 1);

  lua_settop(L, 2);  
  return 2;
}

static int lab_(max)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  luaT_pushudata(L, THLongTensor_new(), torch_LongTensor_id);
  lua_insert(L, 2);
  return lab_(max_)(L);
}

static int lab_(min_)(lua_State *L)
{
  THTensor *values_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THLongTensor *indices_ = luaT_checkudata(L, 2, torch_LongTensor_id);
  THTensor *t = luaT_checkudata(L, 3, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 4, THTensor_(nDimension)(t)))-1;

  THLab_(min)(values_, indices_, t, dimension);
  THLongTensor_add(indices_, 1);

  lua_settop(L, 2);  
  return 2;
}

static int lab_(min)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  luaT_pushudata(L, THLongTensor_new(), torch_LongTensor_id);
  lua_insert(L, 2);
  return lab_(min_)(L);
}

static int lab_(sum_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 3, THTensor_(nDimension)(t)))-1;

  THLab_(sum)(r_, t, dimension);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(sum)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(sum_)(L);
}

static int lab_(prod_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 3, THTensor_(nDimension)(t)))-1;

  THLab_(prod)(r_, t, dimension);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(prod)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(prod_)(L);
}

static int lab_(cumsum_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 3, THTensor_(nDimension)(t)))-1;

  THLab_(cumsum)(r_, t, dimension);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(cumsum)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(cumsum_)(L);
}

static int lab_(cumprod_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 3, THTensor_(nDimension)(t)))-1;

  THLab_(cumprod)(r_, t, dimension);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(cumprod)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(cumprod_)(L);
}

static int lab_(trace)(lua_State *L)
{
  THTensor *t = luaT_checkudata(L, 1, torch_(Tensor_id));
  real trace;

  THLab_(trace)(&trace, t);

  lua_pushnumber(L, trace);  
  return 1;
}

static int lab_(cross_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *a = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *b = luaT_checkudata(L, 3, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 4, -1))-1;

  THLab_(cross)(r_, a, b, dimension);

  lua_settop(L, 1);
  return 1;
}

static int lab_(cross)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(cross_)(L);
}

static int lab_(zeros_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THLongStorage *dimension = lab_checklongargs(L, 2);

  THLab_(zeros)(r_, dimension);

  THLongStorage_free(dimension);
  lua_settop(L, 1);  
  return 1;
}

static int lab_(zeros)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(zeros_)(L);
}

static int lab_(ones_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THLongStorage *dimension = lab_checklongargs(L, 2);

  THLab_(ones)(r_, dimension);

  THLongStorage_free(dimension);
  lua_settop(L, 1);  
  return 1;
}

static int lab_(ones)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(ones_)(L);
}

static int lab_(diag_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  long k = luaL_optnumber(L, 3, 0);

  THLab_(diag)(r_, t, k);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(diag)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(diag_)(L);
}

static int lab_(eye_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  long n = luaL_checknumber(L, 2);
  long m = luaL_optnumber(L, 3, 0);

  THLab_(eye)(r_, n, m);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(eye)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(eye_)(L);
}

static int lab_(range_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  real xmin = luaL_checknumber(L, 2);
  real xmax = luaL_checknumber(L, 3);
  real step = luaL_optnumber(L, 4, 1);

  THLab_(range)(r_, xmin, xmax, step);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(range)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(range_)(L);
}

static int lab_(randperm_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  long n = (long)luaL_checknumber(L, 2);

  THLab_(randperm)(r_, n);
  THTensor_(add)(r_, 1);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(randperm)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(randperm_)(L);
}

static int lab_(reshape_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  THLongStorage *dimension = lab_checklongargs(L, 3);

  THLab_(reshape)(r_, t, dimension);

  THLongStorage_free(dimension);
  lua_settop(L, 1);  
  return 1;
}

static int lab_(reshape)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(reshape_)(L);
}

static int lab_(sort_)(lua_State *L)
{
  THTensor *rt_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THLongTensor *ri_ = luaT_checkudata(L, 2, torch_LongTensor_id);
  THTensor *t = luaT_checkudata(L, 3, torch_(Tensor_id));
  int dimension = luaL_optnumber(L, 4, THTensor_(nDimension)(t))-1;
  int descendingOrder = luaT_optboolean(L, 5, 0);

  THLab_(sort)(rt_, ri_, t, dimension, descendingOrder);
  THLongTensor_add(ri_, 1);

  lua_settop(L, 2);
  return 2;
}

static int lab_(sort)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  luaT_pushudata(L, THLongTensor_new(), torch_LongTensor_id);
  lua_insert(L, 2);
  return lab_(sort_)(L);
}

static int lab_(tril_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int k = (int)(luaL_optnumber(L, 3, 0));

  THLab_(tril)(r_, t, k);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(tril)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(tril_)(L);
}

static int lab_(triu_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int k = (int)(luaL_optnumber(L, 3, 0));

  THLab_(triu)(r_, t, k);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(triu)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(triu_)(L);
}

static int lab_(cat_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *ta = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *tb = luaT_checkudata(L, 3, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 4, 1))-1;

  THLab_(cat)(r_, ta, tb, dimension);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(cat)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(cat_)(L);
}

static int lab_(histc)(lua_State *L)
{
  THTensor *r = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *h = luaT_checkudata(L, 2, torch_(Tensor_id));
  int nbins = luaL_checknumber(L, 3);
  real *h_data = THTensor_(data)(h);

  TH_TENSOR_APPLY(real, r,                                      \
                  if ((*r_data <= nbins) && (*r_data >= 1)) {   \
                    *(h_data + (int)(*r_data) - 1) += 1;        \
                  })
  return 0;
}

static int lab_(convxcorr2)(lua_State *L,char* ktype)
{
  THTensor *r_ = NULL;
  THTensor *image = luaT_checkudata(L,1,torch_(Tensor_id));
  THTensor *kernel = luaT_checkudata(L,2,torch_(Tensor_id));
  int n = lua_gettop(L);
  const char* ctype = "v";
  if (n == 2)
  {
    r_ = THTensor_(new)();
  }
  else if (n == 3)
  {
    if (luaT_isudata(L,3, torch_(Tensor_id)))
    {
      r_ = image;
      image = kernel;
      kernel = luaT_checkudata(L,3,torch_(Tensor_id));
    }
    else if (lua_isstring(L,3))
    {
      r_ = THTensor_(new)();
      ctype = luaL_checkstring(L,3);
    }
    else
    {
      return luaL_error(L, "bad arguments: [result,] source, kernel [, conv type]");
    }
  }
  else if (n == 4)
  {
    r_ = image;
    image = kernel;
    kernel = luaT_checkudata(L,3,torch_(Tensor_id));
    ctype = luaL_checkstring(L,4);
  }
  else
  {
    return luaL_error(L, "bad arguments: [result,] source, kernel [, conv type]");
  }
  if (!r_)
  {
    return luaL_error(L, "oops, bad arguments: [result,] source, kernel [, conv type]");
  }
  else
  {
    luaT_pushudata(L, r_, torch_(Tensor_id));
  }

  char type[2];
  type[0] = ctype[0];
  type[1] = ktype[0];

  if (image->nDimension == 2 && kernel->nDimension == 2)
  {
    THLab_(conv2Dmul)(r_,0.0,1.0,image,kernel,1,1,type);
  }
  else if (image->nDimension == 3 && kernel->nDimension == 3)
  {
    THLab_(conv2Dger)(r_,0.0,1.0,image,kernel,1,1,type);
  }
  else if (image->nDimension == 3 && kernel->nDimension == 4)
  {
    THLab_(conv2Dmv)(r_,0.0,1.0,image,kernel,1,1,type);
  }
  else if (image->nDimension == 2 && kernel->nDimension == 3)
  {
    if (kernel->size[0] > 1)
    {
      long k;
      THTensor *ri = THTensor_(new)();
      THTensor *ker = THTensor_(new)();

      long nInputRows  = image->size[0];
      long nInputCols  = image->size[1];
      long nKernelRows = kernel->size[1];
      long nKernelCols = kernel->size[2];
      long nOutputRows, nOutputCols;

      THArgCheck((nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *type == 'f', 2, "Input image is smaller than kernel");
  
      if (type[0] == 'f') {
	nOutputRows = (nInputRows - 1) * 1 + nKernelRows;
	nOutputCols = (nInputCols - 1) * 1 + nKernelCols;
      } else { // valid
	nOutputRows = (nInputRows - nKernelRows) / 1 + 1;
	nOutputCols = (nInputCols - nKernelCols) / 1 + 1;
      }

      THTensor_(resize3d)(r_,kernel->size[0], nOutputRows, nOutputCols);
      for (k=0; k<kernel->size[0]; k++)
      {
        THTensor_(select)(ker,kernel,0,k);
        THTensor_(select)(ri,r_,0,k);
        THLab_(conv2Dmul)(ri,0.0,1.0,image,ker,1,1,type);
      }
      THTensor_(free)(ri);
      THTensor_(free)(ker);
    } else {
      THTensor *ker = THTensor_(new)();
      THTensor_(select)(ker,kernel,0,0);
      THLab_(conv2Dmul)(r_,0.0,1.0,image,ker,1,1,type);
      THTensor_(free)(ker);
    }
  }
  else if (image->nDimension == 3 && kernel->nDimension == 2)
  {
    if (image->size[0] > 1)
    {
      long k;
      THTensor *ri = THTensor_(new)();
      THTensor *im = THTensor_(new)();

      long nInputRows  = image->size[1];
      long nInputCols  = image->size[2];
      long nKernelRows = kernel->size[0];
      long nKernelCols = kernel->size[1];
      long nOutputRows, nOutputCols;

      THArgCheck((nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *type == 'f', 2, "Input image is smaller than kernel");
  
      if (type[0] == 'f') {
	nOutputRows = (nInputRows - 1) * 1 + nKernelRows;
	nOutputCols = (nInputCols - 1) * 1 + nKernelCols;
      } else { // valid
	nOutputRows = (nInputRows - nKernelRows) / 1 + 1;
	nOutputCols = (nInputCols - nKernelCols) / 1 + 1;
      }
      THTensor_(resize3d)(r_,image->size[0], nOutputRows, nOutputCols);
      for (k=0; k<image->size[0]; k++)
      {
        THTensor_(select)(im, image, 0, k);
        THTensor_(select)(ri,r_,0,k);
        THLab_(conv2Dmul)(ri,0.0,1.0,im,kernel,1,1,type);
      }
      THTensor_(free)(ri);
      THTensor_(free)(im);
    } else {
      THTensor *im = THTensor_(new)();
      THTensor_(select)(im,image,0,0);
      THLab_(conv2Dmul)(r_,0.0,1.0,im,kernel,1,1,type);
      THTensor_(free)(im);
    }
  }
  return 1;
}

static int lab_(conv2)(lua_State *L)
{
  return lab_(convxcorr2)(L,"convolution");
}
static int lab_(xcorr2)(lua_State *L)
{
  return lab_(convxcorr2)(L,"xcorrelation");
}

static int lab_(convxcorr3)(lua_State *L,char* ktype)
{
  THTensor *r_ = NULL;
  THTensor *image = luaT_checkudata(L,1,torch_(Tensor_id));
  THTensor *kernel = luaT_checkudata(L,2,torch_(Tensor_id));
  int n = lua_gettop(L);
  const char* ctype = "v";
  if (n == 2)
  {
    r_ = THTensor_(new)();
  }
  else if (n == 3)
  {
    if (luaT_isudata(L,3, torch_(Tensor_id)))
    {
      r_ = image;
      image = kernel;
      kernel = luaT_checkudata(L,3,torch_(Tensor_id));
    }
    else if (lua_isstring(L,3))
    {
      r_ = THTensor_(new)();
      ctype = luaL_checkstring(L,3);
    }
    else
    {
      return luaL_error(L, "bad arguments: [result,] source, kernel [, conv type]");
    }
  }
  else if (n == 4)
  {
    r_ = image;
    image = kernel;
    kernel = luaT_checkudata(L,3,torch_(Tensor_id));
    ctype = luaL_checkstring(L,4);
  }
  else
  {
    return luaL_error(L, "bad arguments: [result,] source, kernel [, conv type]");
  }
  if (!r_)
  {
    return luaL_error(L, "oops, bad arguments: [result,] source, kernel [, conv type]");
  }
  else
  {
    luaT_pushudata(L, r_, torch_(Tensor_id));
  }

  char type[2];
  type[0] = ctype[0];
  type[1] = ktype[0];

  if (image->nDimension == 3 && kernel->nDimension == 3)
  {
    THLab_(conv3Dmul)(r_,0.0,1.0,image,kernel,1,1,1,type);
  }
  else if (image->nDimension == 4 && kernel->nDimension == 4)
  {
    THLab_(conv3Dger)(r_,0.0,1.0,image,kernel,1,1,1,type);
  }
  else if (image->nDimension == 4 && kernel->nDimension == 5)
  {
    THLab_(conv3Dmv)(r_,0.0,1.0,image,kernel,1,1,1,type);
  }
  else if (image->nDimension == 3 && kernel->nDimension == 4)
  {
    if (kernel->size[0] > 1)
    {
      long k;
      THTensor *ri = THTensor_(new)();
      THTensor *ker = THTensor_(new)();

      long nInputDepth = image->size[0];
      long nInputRows  = image->size[1];
      long nInputCols  = image->size[2];
      long nKernelDepth= kernel->size[1];
      long nKernelRows = kernel->size[2];
      long nKernelCols = kernel->size[3];
      long nOutputDepth, nOutputRows, nOutputCols;

      THArgCheck((nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *type == 'f', 2, "Input image is smaller than kernel");
  
      if (type[0] == 'f') {
	nOutputDepth = (nInputDepth - 1) * 1 + nKernelDepth;
	nOutputRows = (nInputRows - 1) * 1 + nKernelRows;
	nOutputCols = (nInputCols - 1) * 1 + nKernelCols;
      } else { // valid
	nOutputDepth = (nInputDepth - nKernelDepth) / 1 + 1;
	nOutputRows = (nInputRows - nKernelRows) / 1 + 1;
	nOutputCols = (nInputCols - nKernelCols) / 1 + 1;
      }

      THTensor_(resize4d)(r_,kernel->size[0], nOutputDepth, nOutputRows, nOutputCols);
      for (k=0; k<kernel->size[0]; k++)
      {
        THTensor_(select)(ker,kernel,0,k);
        THTensor_(select)(ri,r_,0,k);
        THLab_(conv3Dmul)(ri,0.0,1.0,image,ker,1,1,1,type);
      }
      THTensor_(free)(ri);
      THTensor_(free)(ker);
    } else {
      THTensor *ker = THTensor_(new)();
      THTensor_(select)(ker,kernel,0,0);
      THLab_(conv3Dmul)(r_,0.0,1.0,image,ker,1,1,1,type);
      THTensor_(free)(ker);
    }
  }
  else if (image->nDimension == 4 && kernel->nDimension == 3)
  {
    if (image->size[0] > 1)
    {
      long k;
      THTensor *ri = THTensor_(new)();
      THTensor *im = THTensor_(new)();

      long nInputDepth = image->size[1];
      long nInputRows  = image->size[2];
      long nInputCols  = image->size[3];
      long nKernelDepth= kernel->size[0];
      long nKernelRows = kernel->size[1];
      long nKernelCols = kernel->size[2];
      long nOutputDepth, nOutputRows, nOutputCols;

      THArgCheck((nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *type == 'f', 2, "Input image is smaller than kernel");
  
      if (type[0] == 'f') {
	nOutputDepth = (nInputDepth - 1) * 1 + nKernelDepth;
	nOutputRows = (nInputRows - 1) * 1 + nKernelRows;
	nOutputCols = (nInputCols - 1) * 1 + nKernelCols;
      } else { // valid
	nOutputDepth = (nInputDepth - nKernelDepth) / 1 + 1;
	nOutputRows = (nInputRows - nKernelRows) / 1 + 1;
	nOutputCols = (nInputCols - nKernelCols) / 1 + 1;
      }
      THTensor_(resize4d)(r_,image->size[0], nOutputDepth, nOutputRows, nOutputCols);
      for (k=0; k<image->size[0]; k++)
      {
        THTensor_(select)(im, image, 0, k);
        THTensor_(select)(ri,r_,0,k);
        THLab_(conv3Dmul)(ri,0.0,1.0,im,kernel,1,1,1,type);
      }
      THTensor_(free)(ri);
      THTensor_(free)(im);
    } else {
      THTensor *im = THTensor_(new)();
      THTensor_(select)(im,image,0,0);
      THLab_(conv3Dmul)(r_,0.0,1.0,im,kernel,1,1,1,type);
      THTensor_(free)(im);
    }
  }
  return 1;
}

static int lab_(conv3)(lua_State *L)
{
  return lab_(convxcorr3)(L,"convolution");
}
static int lab_(xcorr3)(lua_State *L)
{
  return lab_(convxcorr3)(L,"xcorrelation");
}

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

static int lab_(mean_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 3, THTensor_(nDimension)(t)))-1;

  THLab_(mean)(r_, t, dimension);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(mean)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(mean_)(L);
}

static int lab_(std_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 3, THTensor_(nDimension)(t)))-1;
  int flag = luaT_optboolean(L, 4, 0);

  THLab_(std)(r_, t, dimension, flag);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(std)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(std_)(L);
}

static int lab_(var_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 3, THTensor_(nDimension)(t)))-1;
  int flag = luaT_optboolean(L, 4, 0);

  THLab_(var)(r_, t, dimension, flag);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(var)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(var_)(L);
}

static int lab_(norm)(lua_State *L)
{
  THTensor *t = luaT_checkudata(L, 1, torch_(Tensor_id));
  real value = luaL_optnumber(L, 2, 2);
  real norm;

  THLab_(norm)(&norm, t, value);
  lua_pushnumber(L, norm);

  return 1;
}

static int lab_(dist)(lua_State *L)
{
  THTensor *t1 = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t2 = luaT_checkudata(L, 2, torch_(Tensor_id));
  real value = luaL_optnumber(L, 3, 2);
  real dist;

  THLab_(dist)(&dist, t1, t2, value);
  lua_pushnumber(L, dist);

  return 1;
}

static int lab_(linspace_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  real a = luaL_checknumber(L, 2);
  real b = luaL_checknumber(L, 3);
  long n = luaL_optnumber(L, 4, 100);

  THLab_(linspace)(r_, a, b, n);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(linspace)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(linspace_)(L);
}

static int lab_(logspace_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  real a = luaL_checknumber(L, 2);
  real b = luaL_checknumber(L, 3);
  long n = luaL_optnumber(L, 4, 100);

  THLab_(logspace)(r_, a, b, n);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(logspace)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(logspace_)(L);
}

static int lab_(rand_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THLongStorage *dimension = lab_checklongargs(L, 2);

  THLab_(rand)(r_, dimension);

  THLongStorage_free(dimension);
  lua_settop(L, 1);  
  return 1;
}

static int lab_(rand)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(rand_)(L);
}

static int lab_(randn_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THLongStorage *dimension = lab_checklongargs(L, 2);

  THLab_(randn)(r_, dimension);

  THLongStorage_free(dimension);
  lua_settop(L, 1);  
  return 1;
}

static int lab_(randn)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(randn_)(L);
}

#define LAB_IMPLEMENT_MATH_FUNC(NAME)                         \
  static int lab_(NAME##_)(lua_State *L)                      \
  {                                                           \
    THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));  \
    THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));   \
                                                              \
    THLab_(NAME)(r_, t);                                      \
                                                              \
    lua_settop(L, 1);                                         \
    return 1;                                                 \
  }                                                           \
                                                              \
  static int lab_(NAME)(lua_State *L)                         \
  {                                                           \
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));   \
    lua_insert(L, 1);                                         \
    return lab_(NAME##_)(L);                                  \
  }                                                           \
                                                              \

LAB_IMPLEMENT_MATH_FUNC(log)
LAB_IMPLEMENT_MATH_FUNC(log1p)
LAB_IMPLEMENT_MATH_FUNC(exp)
LAB_IMPLEMENT_MATH_FUNC(cos)
LAB_IMPLEMENT_MATH_FUNC(acos)
LAB_IMPLEMENT_MATH_FUNC(cosh)
LAB_IMPLEMENT_MATH_FUNC(sin)
LAB_IMPLEMENT_MATH_FUNC(asin)
LAB_IMPLEMENT_MATH_FUNC(sinh)
LAB_IMPLEMENT_MATH_FUNC(tan)
LAB_IMPLEMENT_MATH_FUNC(atan)
LAB_IMPLEMENT_MATH_FUNC(tanh)
LAB_IMPLEMENT_MATH_FUNC(sqrt)
LAB_IMPLEMENT_MATH_FUNC(ceil)
LAB_IMPLEMENT_MATH_FUNC(floor)
LAB_IMPLEMENT_MATH_FUNC(abs)

static int lab_(pow_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  real value = (real)luaL_checknumber(L, 3);
  THLab_(pow)(r_, t, value);
  lua_settop(L, 1);
  return 1;
}

static int lab_(pow)(lua_State *L)
{
  luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
  lua_insert(L, 1);
  return lab_(pow_)(L);
}

#endif

static const struct luaL_Reg lab_(stuff__) [] = {
  {"numel", lab_(numel)},
  {"max_", lab_(max_)},
  {"max", lab_(max)},
  {"min_", lab_(min_)},
  {"min", lab_(min)},
  {"sum_", lab_(sum_)},
  {"sum", lab_(sum)},
  {"prod_", lab_(prod_)},
  {"prod", lab_(prod)},
  {"cumsum_", lab_(cumsum_)},
  {"cumsum", lab_(cumsum)},
  {"cumprod_", lab_(cumprod_)},
  {"cumprod", lab_(cumprod)},
  {"trace", lab_(trace)},
  {"cross_", lab_(cross_)},
  {"cross", lab_(cross)},
  {"zeros_", lab_(zeros_)},
  {"zeros", lab_(zeros)},
  {"ones_", lab_(ones_)},
  {"ones", lab_(ones)},
  {"diag_", lab_(diag_)},
  {"diag", lab_(diag)},
  {"eye_", lab_(eye_)},
  {"eye", lab_(eye)},
  {"range_", lab_(range_)},
  {"range", lab_(range)},
  {"randperm_", lab_(randperm_)},
  {"randperm", lab_(randperm)},
  {"reshape_", lab_(reshape_)},
  {"reshape", lab_(reshape)},
  {"sort_", lab_(sort_)},
  {"sort", lab_(sort)},
  {"tril_", lab_(tril_)},
  {"tril", lab_(tril)},
  {"triu_", lab_(triu_)},
  {"triu", lab_(triu)},
  {"_histc", lab_(histc)},
  {"cat_", lab_(cat_)},
  {"cat", lab_(cat)},
  {"conv2", lab_(conv2)},
  {"xcorr2", lab_(xcorr2)},
  {"conv3", lab_(conv3)},
  {"xcorr3", lab_(xcorr3)},
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"log_", lab_(log_)},
  {"log", lab_(log)},
  {"log1p_", lab_(log1p_)},
  {"log1p", lab_(log1p)},
  {"exp_", lab_(exp_)},
  {"exp", lab_(exp)},
  {"cos_", lab_(cos_)},
  {"cos", lab_(cos)},
  {"acos_", lab_(acos_)},
  {"acos", lab_(acos)},
  {"cosh_", lab_(cosh_)},
  {"cosh", lab_(cosh)},
  {"sin_", lab_(sin_)},
  {"sin", lab_(sin)},
  {"asin_", lab_(asin_)},
  {"asin", lab_(asin)},
  {"sinh_", lab_(sinh_)},
  {"sinh", lab_(sinh)},
  {"tan_", lab_(tan_)},
  {"tan", lab_(tan)},
  {"atan_", lab_(atan_)},
  {"atan", lab_(atan)},
  {"tanh_", lab_(tanh_)},
  {"tanh", lab_(tanh)},
  {"pow_", lab_(pow_)},
  {"pow", lab_(pow)},
  {"sqrt_", lab_(sqrt_)},
  {"sqrt", lab_(sqrt)},
  {"ceil_", lab_(ceil_)},
  {"ceil", lab_(ceil)},
  {"floor_", lab_(floor_)},
  {"floor", lab_(floor)},
  {"abs_", lab_(abs_)},
  {"abs", lab_(abs)},
  {"mean_", lab_(mean_)},
  {"mean", lab_(mean)},
  {"std_", lab_(std_)},
  {"std", lab_(std)},
  {"var_", lab_(var_)},
  {"var", lab_(var)},
  {"norm", lab_(norm)},
  {"dist", lab_(dist)},
  {"linspace_", lab_(linspace_)},
  {"linspace", lab_(linspace)},
  {"logspace_", lab_(logspace_)},
  {"logspace", lab_(logspace)},
  {"rand_", lab_(rand_)},
  {"rand", lab_(rand)},
  {"randn_", lab_(randn_)},
  {"randn", lab_(randn)},
#endif
  {NULL, NULL}
};

void lab_(init)(lua_State *L)
{
  torch_(Tensor_id) = luaT_checktypename2id(L, torch_string_(Tensor));
  torch_LongStorage_id = luaT_checktypename2id(L, "torch.LongStorage");

  /* register everything into the "lab" field of the tensor metaclass */
  luaT_pushmetaclass(L, torch_(Tensor_id));
  lua_pushstring(L, "lab");
  lua_newtable(L);
  luaL_register(L, NULL, lab_(stuff__));
  lua_rawset(L, -3);
  lua_pop(L, 1);

/*  luaT_registeratid(L, lab_(stuff__), torch_(Tensor_id)); */
/*  luaL_register(L, NULL, lab_(stuff__)); */  
}

#endif
