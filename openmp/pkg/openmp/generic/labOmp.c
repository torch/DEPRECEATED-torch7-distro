#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/labOmp.c"
#else

static int labOmp_(convxcorr2omp)(lua_State *L, char* ktype)
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
    THOmpLab_(conv2Dger)(r_,0.0,1.0,image,kernel,1,1,type);
  }
  else if (image->nDimension == 3 && kernel->nDimension == 4)
  {
    THOmpLab_(conv2Dmv)(r_,0.0,1.0,image,kernel,1,1,type);
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

static int labOmp_(conv2omp)(lua_State *L)
{
  return labOmp_(convxcorr2omp)(L,"convolution");
}
static int labOmp_(xcorr2omp)(lua_State *L)
{
  return labOmp_(convxcorr2omp)(L,"xcorrelation");
}



static const struct luaL_Reg labOmp_(stuff__) [] = {
  {"conv2omp", labOmp_(conv2omp)},
  {"xcorr2omp", labOmp_(xcorr2omp)},
  {NULL,NULL}
};

void labOmp_(init)(lua_State *L)
{
  torch_(Tensor_id) = luaT_checktypename2id(L, torch_string_(Tensor));
  torch_LongStorage_id = luaT_checktypename2id(L, "torch.LongStorage");

  /* register everything into the field of the tensor metaclass */
  luaT_pushmetaclass(L, torch_(Tensor_id));
  lua_getfield(L,-1,"lab");
  luaL_register(L, NULL, labOmp_(stuff__));
  lua_pop(L, 1);
}

#endif
