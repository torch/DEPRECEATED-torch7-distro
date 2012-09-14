struct hardtanhupdateOutput_functor
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

static int cunn_HardTanh_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);

  THCudaTensor_resizeAs(output, input);

  thrust::device_ptr<float> output_data(THCudaTensor_data(output));
  thrust::device_ptr<float> input_data(THCudaTensor_data(input));
  thrust::transform(input_data, input_data+size, output_data, hardtanhupdateOutput_functor());

  THCudaTensor_free(input);
  return 1;
}

struct hardtanhupdateGradInput_functor
{
  __host__ __device__ float operator()(const float& input, const float& gradOutput) const
  {
    if(input < -1 || input > 1)
      return 0;
    else
      return gradOutput;
  }
};

static int cunn_HardTanh_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);
  gradOutput = THCudaTensor_newContiguous(gradOutput);

  THCudaTensor_resizeAs(gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(input));
  thrust::device_ptr<float> gradOutput_data(THCudaTensor_data(gradOutput));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(gradInput));
  thrust::transform(input_data, input_data+size, gradOutput_data, gradInput_data, hardtanhupdateGradInput_functor());

  THCudaTensor_free(gradOutput);
  THCudaTensor_free(input);
  return 1;
}

static const struct luaL_Reg cunn_HardTanh__ [] = {
  {"HardTanh_updateOutput", cunn_HardTanh_updateOutput},
  {"HardTanh_updateGradInput", cunn_HardTanh_updateGradInput},
  {NULL, NULL}
};

static void cunn_HardTanh_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_HardTanh__, "nn");
  lua_pop(L,1);
}
