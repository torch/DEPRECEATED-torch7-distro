struct tanhupdateOutput_functor
{
  __host__ __device__ float operator()(const float& input) const
  {
    return tanh(input);
  }
};

static int cunn_Tanh_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);

  THCudaTensor_resizeAs(output, input);

  thrust::device_ptr<float> output_data(THCudaTensor_data(output));
  thrust::device_ptr<float> input_data(THCudaTensor_data(input));
  thrust::transform(input_data, input_data+size, output_data, tanhupdateOutput_functor());

  THCudaTensor_free(input);
  return 1;
}

struct tanhupdateGradInput_functor
{
  __host__ __device__ float operator()(const float& output, const float& gradOutput) const
  {
    return gradOutput * (1-output*output);
  }
};

static int cunn_Tanh_updateGradInput(lua_State *L)
{
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long size = THCudaTensor_nElement(output);

  gradOutput = THCudaTensor_newContiguous(gradOutput);

  THCudaTensor_resizeAs(gradInput, output);

  thrust::device_ptr<float> output_data(THCudaTensor_data(output));
  thrust::device_ptr<float> gradOutput_data(THCudaTensor_data(gradOutput));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(gradInput));
  thrust::transform(output_data, output_data+size, gradOutput_data, gradInput_data, tanhupdateGradInput_functor());

  THCudaTensor_free(gradOutput);
  return 1;
}

static const struct luaL_Reg cunn_Tanh__ [] = {
  {"Tanh_updateOutput", cunn_Tanh_updateOutput},
  {"Tanh_updateGradInput", cunn_Tanh_updateGradInput},
  {NULL, NULL}
};

static void cunn_Tanh_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Tanh__, "nn");
  lua_pop(L,1);
}
