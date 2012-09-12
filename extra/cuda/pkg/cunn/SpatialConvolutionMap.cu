static int cunn_SpatialConvolutionMap_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)
    luaT_checkudata(L, 2, "torch.CudaTensor");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THCudaTensor *weight = (THCudaTensor*)
    luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)
    luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)
    luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *connTableRev = (THCudaTensor*)
    luaT_getfieldcheckudata(L, 1, "connTableRev", "torch.CudaTensor");
  luaL_argcheck(L, connTableRev->nDimension == 3, 2, 
                "Reverse table not generated (is table fixed fanin?)");
  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor is expected");

  int dimw = 2;
  int dimh = 1;

  // long nOutputPlane = weight->size[0];
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  long kW           = weight->size[2];
  long kH           = weight->size[1];
  long inputWidth   = input->size[dimw];
  long inputHeight  = input->size[dimh];
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;
  long fanin        = weight->size[0] / nOutputPlane;
  THCudaTensor_resize3d(output, nOutputPlane, outputHeight, outputWidth);
  /* add bias first */
  long k;
  THCudaTensor *outputPlane = THCudaTensor_new();
  for(k=0; k<nOutputPlane; k++) {
    THCudaTensor_select(outputPlane, output, 0, k);
    THCudaTensor_fill(outputPlane, THCudaTensor_get1d(bias, k));
  }
  THCudaTensor_free(outputPlane);
  /* do convolutions */
  THCudaTensor_conv2Dmap(output, input, weight, dW, dH, 
                         connTableRev,fanin);
  return 1;
}

static const struct luaL_Reg cunn_SpatialConvolutionMap__ [] = {
  {"SpatialConvolutionMap_updateOutput", cunn_SpatialConvolutionMap_updateOutput},

  {NULL, NULL}
};

static void cunn_SpatialConvolutionMap_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialConvolutionMap__, "nn");
  lua_pop(L,1);
}
