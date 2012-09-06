struct squareupdateOutput_functor
{
  __host__ __device__ float operator()(const float& input) const
  {
    return input*input;
  }
};

static int cunn_Square_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, torch_CudaTensor_id);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", torch_CudaTensor_id);
  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);

  THCudaTensor_resizeAs(output, input);

  thrust::device_ptr<float> output_data(THCudaTensor_data(output));
  thrust::device_ptr<float> input_data(THCudaTensor_data(input));
  thrust::transform(input_data, input_data+size, output_data, squareupdateOutput_functor());

  THCudaTensor_free(input);
  return 1;
}

struct squareupdateGradInput_functor
{
  __host__ __device__ float operator()(const float& input, const float& gradOutput) const
  {
    return 2.0 * gradOutput * input;
  }
};

static int cunn_Square_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 3, torch_CudaTensor_id);
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, torch_CudaTensor_id);
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", torch_CudaTensor_id);
  long size = THCudaTensor_nElement(input);

  gradOutput = THCudaTensor_newContiguous(gradOutput);
  THCudaTensor_resizeAs(gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(input));
  thrust::device_ptr<float> gradOutput_data(THCudaTensor_data(gradOutput));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(gradInput));
  thrust::transform(input_data, input_data+size, gradOutput_data, gradInput_data, squareupdateGradInput_functor());

  THCudaTensor_free(gradOutput);
  return 1;
}

static const struct luaL_Reg cunn_Square__ [] = {
  {"Square_updateOutput", cunn_Square_updateOutput},
  {"Square_updateGradInput", cunn_Square_updateGradInput},
  {NULL, NULL}
};

static void cunn_Square_init(lua_State *L)
{
  luaT_pushmetaclass(L, torch_CudaTensor_id);
  luaT_registeratname(L, cunn_Square__, "nn");
  lua_pop(L,1);
}
