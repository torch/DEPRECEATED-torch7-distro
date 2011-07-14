struct hardtanhforward_functor
{
  __host__ __device__ float operator()(const float& input) const
  {
    if(input < -1)
      return -1;
    else if(input <= 1)
      return input;
    else
      return 1;
  }
};

static int cunn_HardTanh_forward(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, torch_CudaTensor_id);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", torch_CudaTensor_id);
  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);

  THCudaTensor_resizeAs(output, input);

  thrust::device_ptr<float> output_data(THCudaTensor_data(output));
  thrust::device_ptr<float> input_data(THCudaTensor_data(input));
  thrust::transform(input_data, input_data+size, output_data, hardtanhforward_functor());

  THCudaTensor_free(input);
  return 1;
}

struct hardtanhbackward_functor
{
  __host__ __device__ float operator()(const float& input, const float& gradOutput) const
  {
    if(input < -1 || input > 1)
      return 0;
    else
      return gradOutput;
  }
};

static int cunn_HardTanh_backward(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, torch_CudaTensor_id);
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, torch_CudaTensor_id);
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", torch_CudaTensor_id);
  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);
  gradOutput = THCudaTensor_newContiguous(gradOutput);

  THCudaTensor_resizeAs(gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(input));
  thrust::device_ptr<float> gradOutput_data(THCudaTensor_data(gradOutput));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(gradInput));
  thrust::transform(input_data, input_data+size, gradOutput_data, gradInput_data, hardtanhbackward_functor());

  THCudaTensor_free(gradOutput);
  THCudaTensor_free(input);
  return 1;
}

static const struct luaL_Reg cunn_HardTanh__ [] = {
  {"HardTanh_forward", cunn_HardTanh_forward},
  {"HardTanh_backward", cunn_HardTanh_backward},
  {NULL, NULL}
};

static void cunn_HardTanh_init(lua_State *L)
{
  luaT_pushmetaclass(L, torch_CudaTensor_id);
  luaT_registeratname(L, cunn_HardTanh__, "nn");
  lua_pop(L,1);
}
