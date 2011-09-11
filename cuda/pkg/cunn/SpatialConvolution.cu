
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
  long kW           = weight->size[3];
  long kH           = weight->size[2];
  long inputWidth   = input->size[2];
  long inputHeight  = input->size[1];
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;

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
  THCudaTensor_conv2Dmv(output, 1.0, input, weight, dH, dW, "vx");

  return 1;
}

static int cunn_SpatialConvolution_backward(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, torch_CudaTensor_id);
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, torch_CudaTensor_id);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  luaL_argcheck(L, dW == 1, 1, "dW must be 1 (this will be fixed soon)");
  luaL_argcheck(L, dH == 1, 1, "dH must be 1 (this will be fixed soon)");

  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", torch_CudaTensor_id);
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", torch_CudaTensor_id);

  THArgCheck(nOutputPlane == gradOutput->size[0], 1, "Number of output features is not equal to nOutputPlane");

  /* gradient to input */
  THCudaTensor *tweight = THCudaTensor_newTranspose(weight,0,1);
  THCudaTensor_conv2Dmv(gradInput, 0.0, gradOutput, tweight, dH, dW, "fc");
  THCudaTensor_free(tweight);

  return 1;
}

static int cunn_SpatialConvolution_accGradParameters(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, torch_CudaTensor_id);
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, torch_CudaTensor_id);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  luaL_argcheck(L, dW == 1, 1, "dW must be 1 (this will be fixed soon)");
  luaL_argcheck(L, dH == 1, 1, "dH must be 1 (this will be fixed soon)");

  THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", torch_CudaTensor_id);
  THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", torch_CudaTensor_id);

  THArgCheck(nOutputPlane == gradOutput->size[0], 1, "Number of output features is not equal to nOutputPlane");

  /* gradient to bias */
  long k;
  THCudaTensor *gradOutSlice = THCudaTensor_new();
  for(k = 0; k < nOutputPlane; k++) {
    THCudaTensor_select(gradOutSlice, gradOutput, 0, k);
    float sum = THCudaTensor_sum(gradOutSlice);
    THCudaTensor_set1d(gradBias, k, THCudaTensor_get1d(gradBias, k) + sum);
  }
  THCudaTensor_free(gradOutSlice);

  /* gradient to kernels */
  THCudaTensor_conv2DRevger(gradWeight, 1.0, input, gradOutput, dH, dW);

  return 0;
}

static const struct luaL_Reg cunn_SpatialConvolution__ [] = {
  {"SpatialConvolution_forward", cunn_SpatialConvolution_forward},
  {"SpatialConvolution_backward", cunn_SpatialConvolution_backward},
  {"SpatialConvolution_accGradParameters", cunn_SpatialConvolution_accGradParameters},
  {NULL, NULL}
};

static void cunn_SpatialConvolution_init(lua_State *L)
{
  luaT_pushmetaclass(L, torch_CudaTensor_id);
  luaT_registeratname(L, cunn_SpatialConvolution__, "nn");
  lua_pop(L,1);
}
