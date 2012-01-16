#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TensorConv.c"
#else

static int torch_(convxcorr2)(lua_State *L,char* ktype)
{
  THTensor *r_ = NULL;
  THTensor *image = luaT_checkudata(L,1,torch_(Tensor_id));
  THTensor *kernel = luaT_checkudata(L,2,torch_(Tensor_id));
  int n = lua_gettop(L);
  const char* ctype = "v";
  if (n == 2)
  {
    r_ = THTensor_(new)();
    luaT_pushudata(L, r_, torch_(Tensor_id));
  }
  else if (n == 3)
  {
    if (luaT_isudata(L,3, torch_(Tensor_id)))
    {
      r_ = image;
      image = kernel;
      kernel = luaT_checkudata(L,3,torch_(Tensor_id));
      lua_settop(L,1);
    }
    else if (lua_isstring(L,3))
    {
      r_ = THTensor_(new)();
      ctype = luaL_checkstring(L,3);
      luaT_pushudata(L, r_, torch_(Tensor_id));
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
    lua_settop(L,1);
  }
  else
  {
    return luaL_error(L, "bad arguments: [result,] source, kernel [, conv type]");
  }
  if (!r_)
  {
    return luaL_error(L, "oops, bad arguments: [result,] source, kernel [, conv type]");
  }
/*   else */
/*   { */
/*     //luaT_pushudata(L, r_, torch_(Tensor_id)); */
/*   } */

  char type[2];
  type[0] = ctype[0];
  type[1] = ktype[0];

  if (image->nDimension == 2 && kernel->nDimension == 2)
  {
    THTensor_(conv2Dmul)(r_,0.0,1.0,image,kernel,1,1,type);
  }
  else if (image->nDimension == 3 && kernel->nDimension == 3)
  {
    THTensor_(conv2Dger)(r_,0.0,1.0,image,kernel,1,1,type);
  }
  else if (image->nDimension == 3 && kernel->nDimension == 4)
  {
    THTensor_(conv2Dmv)(r_,0.0,1.0,image,kernel,1,1,type);
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
        THTensor_(conv2Dmul)(ri,0.0,1.0,image,ker,1,1,type);
      }
      THTensor_(free)(ri);
      THTensor_(free)(ker);
    } else {
      THTensor *ker = THTensor_(new)();
      THTensor_(select)(ker,kernel,0,0);
      THTensor_(conv2Dmul)(r_,0.0,1.0,image,ker,1,1,type);
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
        THTensor_(conv2Dmul)(ri,0.0,1.0,im,kernel,1,1,type);
      }
      THTensor_(free)(ri);
      THTensor_(free)(im);
    } else {
      THTensor *im = THTensor_(new)();
      THTensor_(select)(im,image,0,0);
      THTensor_(conv2Dmul)(r_,0.0,1.0,im,kernel,1,1,type);
      THTensor_(free)(im);
    }
  }
  return 1;
}

static int torch_(conv2)(lua_State *L)
{
  return torch_(convxcorr2)(L,"convolution");
}
static int torch_(xcorr2)(lua_State *L)
{
  return torch_(convxcorr2)(L,"xcorrelation");
}

static int torch_(convxcorr3)(lua_State *L,char* ktype)
{
  THTensor *r_ = NULL;
  THTensor *image = luaT_checkudata(L,1,torch_(Tensor_id));
  THTensor *kernel = luaT_checkudata(L,2,torch_(Tensor_id));
  int n = lua_gettop(L);
  const char* ctype = "v";
  if (n == 2)
  {
    r_ = THTensor_(new)();
    luaT_pushudata(L, r_, torch_(Tensor_id));    
  }
  else if (n == 3)
  {
    if (luaT_isudata(L,3, torch_(Tensor_id)))
    {
      r_ = image;
      image = kernel;
      kernel = luaT_checkudata(L,3,torch_(Tensor_id));
      lua_settop(L,1);
    }
    else if (lua_isstring(L,3))
    {
      r_ = THTensor_(new)();
      ctype = luaL_checkstring(L,3);
      luaT_pushudata(L, r_, torch_(Tensor_id));
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
    lua_settop(L,1);
  }
  else
  {
    return luaL_error(L, "bad arguments: [result,] source, kernel [, conv type]");
  }
  if (!r_)
  {
    return luaL_error(L, "oops, bad arguments: [result,] source, kernel [, conv type]");
  }
/*   else */
/*   { */
/*     luaT_pushudata(L, r_, torch_(Tensor_id)); */
/*   } */

  char type[2];
  type[0] = ctype[0];
  type[1] = ktype[0];

  if (image->nDimension == 3 && kernel->nDimension == 3)
  {
    THTensor_(conv3Dmul)(r_,0.0,1.0,image,kernel,1,1,1,type);
  }
  else if (image->nDimension == 4 && kernel->nDimension == 4)
  {
    THTensor_(conv3Dger)(r_,0.0,1.0,image,kernel,1,1,1,type);
  }
  else if (image->nDimension == 4 && kernel->nDimension == 5)
  {
    THTensor_(conv3Dmv)(r_,0.0,1.0,image,kernel,1,1,1,type);
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
        THTensor_(conv3Dmul)(ri,0.0,1.0,image,ker,1,1,1,type);
      }
      THTensor_(free)(ri);
      THTensor_(free)(ker);
    } else {
      THTensor *ker = THTensor_(new)();
      THTensor_(select)(ker,kernel,0,0);
      THTensor_(conv3Dmul)(r_,0.0,1.0,image,ker,1,1,1,type);
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
        THTensor_(conv3Dmul)(ri,0.0,1.0,im,kernel,1,1,1,type);
      }
      THTensor_(free)(ri);
      THTensor_(free)(im);
    } else {
      THTensor *im = THTensor_(new)();
      THTensor_(select)(im,image,0,0);
      THTensor_(conv3Dmul)(r_,0.0,1.0,im,kernel,1,1,1,type);
      THTensor_(free)(im);
    }
  }
  return 1;
}

static int torch_(conv3)(lua_State *L)
{
  return torch_(convxcorr3)(L,"convolution");
}
static int torch_(xcorr3)(lua_State *L)
{
  return torch_(convxcorr3)(L,"xcorrelation");
}

static const struct luaL_Reg torch_(conv__) [] = {
  {"conv2", torch_(conv2)},
  {"xcorr2", torch_(xcorr2)},
  {"conv3", torch_(conv3)},
  {"xcorr3", torch_(xcorr3)},
  {NULL, NULL}
};

void torch_(conv_init)(lua_State *L)
{
  torch_(Tensor_id) = luaT_checktypename2id(L, torch_string_(Tensor));

  /* register everything into the "torch" field of the tensor metaclass */
  luaT_pushmetaclass(L, torch_(Tensor_id));
  lua_pushstring(L, "torch");
  lua_rawget(L, -2);
  luaL_register(L, NULL, torch_(conv__));
  lua_pop(L, 2);
}

#endif

