#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/labOmp.c"
#else

static int labOmp_(conv2omp)(lua_State *L)
{
  THTensor *r_ = NULL;
  THTensor *image = luaT_checkudata(L,1,torch_(Tensor_id));
  THTensor *kernel = luaT_checkudata(L,2,torch_(Tensor_id));
  int n = lua_gettop(L);
  const char* type = "v";
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
      type = luaL_checkstring(L,3);
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
    type = luaL_checkstring(L,4);
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

  if (image->nDimension == 2 && kernel->nDimension == 2)
  {
    THLab_(conv2Dmul)(r_,0.0,image,kernel,1,1,type);
  }
  else if (image->nDimension == 3 && kernel->nDimension == 3)
  {
    THOmpLab_(conv2Dger)(r_,0.0,image,kernel,1,1,type);
  }
  else if (image->nDimension == 3 && kernel->nDimension == 4)
  {
    THOmpLab_(conv2Dmv)(r_,0.0,image,kernel,1,1,type);
  }
  else if (image->nDimension == 2 && kernel->nDimension == 3)
  {
    if (kernel->size[0] > 1)
    {
      long k;
      THTensor *ri = THTensor_(new)();
      THTensor *ker = THTensor_(new)();
      THTensor_(select)(ker,kernel,0,0);
      THLab_(conv2Dmul)(ri,0.0,image,ker,1,1,type);
      THTensor_(resize3d)(r_,kernel->size[0], ri->size[0], ri->size[1]);
      THTensor_(select)(ker,r_,0,0);
      THTensor_(copy)(ker,ri);
      for (k=1; k<kernel->size[0]; k++)
      {
	THTensor_(select)(ker,kernel,0,k);
	THTensor_(select)(ri,r_,0,k);
	THLab_(conv2Dmul)(ri,0.0,image,ker,1,1,type);
      }
      THTensor_(free)(ri);
      THTensor_(free)(ker);
    } else {
      THTensor *ker = THTensor_(new)();
      THTensor_(select)(ker,kernel,0,0);
      THLab_(conv2Dmul)(r_,0.0,image,ker,1,1,type);
    }
  }
  else if (image->nDimension == 3 && kernel->nDimension == 2)
  {
    if (image->size[0] > 1)
    {
      long k;
      THTensor *ri = THTensor_(new)();
      THTensor *im = THTensor_(new)();
      THTensor_(select)(im,image,0,0);
      THLab_(conv2Dmul)(ri,0.0,im,kernel,1,1,type);
      THTensor_(resize3d)(r_,image->size[0], ri->size[0], ri->size[1]);
      THTensor_(select)(im,r_,0,0);
      THTensor_(copy)(im,ri);
      for (k=1; k<image->size[0]; k++)
      {
	THTensor_(select)(im, image, 0, k);
	THTensor_(select)(ri,r_,0,k);
	THLab_(conv2Dmul)(ri,0.0,im,kernel,1,1,type);
      }
      THTensor_(free)(ri);
      THTensor_(free)(im);
    } else {
      THTensor *im = THTensor_(new)();
      THTensor_(select)(im,image,0,0);
      THLab_(conv2Dmul)(r_,0.0,im,kernel,1,1,type);
    }
  }
  return 1;
}

static const struct luaL_Reg labOmp_(stuff__) [] = {
  {"conv2omp", labOmp_(conv2omp)},
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
