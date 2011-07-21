static int cunn_SpatialConvolution_forward(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, torch_CudaTensor_id);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", torch_CudaTensor_id);
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", torch_CudaTensor_id);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", torch_CudaTensor_id);

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");

  long nOutputPlane = weight->size[0];
  long nInputPlane  = weight->size[1];
  long kW           = weight->size[3];
  long kH           = weight->size[2];
  long inputWidth   = input->size[2];
  long inputHeight  = input->size[1];
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;

  THCudaTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);

  /* add bias first */
  long k;
  THTensor *outputPlane = THTensor_(new)();
  for(k=0; k<nOutputPlane; k++) {
    THCudaTensor_select(outputPlane, output, 0, k);
    THCudaTensor_copy(outputPlane, THCudaTensor_(get1d)(bias, i));
  }
  THCudaTensor_free(outputPlane);

  /* do convolutions */
  THCudaTensor_conv2Dmv(output, 1.0, input, weight, dH, dW, "vx");

  return 1;
}

static int cunn_SpatialConvolution_backward(lua_State *L)
{
  THCudaTensor *input = luaT_checkudata(L, 2, torch_CudaTensor_id);
  THCudaTensor *gradOutput = luaT_checkudata(L, 3, torch_CudaTensor_id);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THCudaTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_CudaTensor_id);
  THCudaTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_CudaTensor_id);
  THCudaTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_CudaTensor_id);
  THCudaTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_CudaTensor_id);

  THArgCheck(nOutputPlane == gradOutput->size[0], 1, "Number of output features is not equal to nOutputPlane");

  long k;

  /* gradient to bias */
  real *gradBias_data = THCudaTensor_(data)(gradBias);
  THCudaTensor *gradOutSlice = THCudaTensor_(new)();
  for(k = 0; k < nOutputPlane; k++)
    {
      THCudaTensor_select(gradOutSlice, gradOutput, 0, k);
      gradBias_data[k] += THCudaTensor_sum(gradOutSlice);
    }
  THCudaTensor_free(gradOutSlice);

  /* gradient to kernels */
  THCudaTensor_conv2DRevger(gradWeight, 1.0, input, gradOutput, dH, dW);

  /* gradient to input */
  THCudaTensor *tweight = THCudaTensor_newTranspose(weight,0,1);
  THCudaTensor_conv2Dmv(gradInput, 0.0, gradOutput, tweight, dH, dW, "fx");
  THCudaTensor_free(tweight);

  return 1;
}

static const struct luaL_Reg cunn_SpatialConvolution__ [] = {
  {"SpatialConvolution_forward", cunn_SpatialConvolution_forward},
  {"SpatialConvolution_backward", cunn_SpatialConvolution_backward},
  {NULL, NULL}
};

static void cunn_SpatialConvolution_init(lua_State *L)
{
  luaT_pushmetaclass(L, torch_CudaTensor_id);
  luaT_registeratname(L, cunn_SpatialConvolution__, "nn");
  lua_pop(L,1);
}
