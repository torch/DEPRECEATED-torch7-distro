#define MINUS_LOG_THRESHOLD -18.42

struct expm_functor
{
  float value;

  expm_functor(float value_) : value(value_) {}

  __host__ __device__ float operator()(const float& x_) const
  {
    float x = value-x_;

# define A0   (1.0)
# define A1   (0.125)
# define A2   (0.0078125)
# define A3   (0.00032552083)
# define A4   (1.0172526e-5)
    if (x < 13.0)
    {
      float y;
      y = A0+x*(A1+x*(A2+x*(A3+x*A4)));
      y *= y;
      y *= y;
      y *= y;
      y = 1/y;
      return y;
    }
    return 0;
# undef A0
# undef A1
# undef A2
# undef A3
# undef A4
  }
};

static int cunn_TemporalLogSoftMax_forward(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, torch_CudaTensor_id);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", torch_CudaTensor_id);
  long size = THCudaTensor_size(input, 1);

  if(input->nDimension != 2)
    luaL_error(L, "input: invalid number of dimension (expected 2)");

  input = THCudaTensor_newContiguous(input);
  THCudaTensor_resizeAs(output, input);

  thrust::device_ptr<float> output_data(THCudaTensor_data(output));
  thrust::device_ptr<float> input_data(THCudaTensor_data(input));
  for(long t = 0; t < input->size[0]; t++)
  {
    float maxInput = thrust::reduce(input_data+t*size, input_data+(t+1)*size, -THInf, thrust::maximum<float>());
    float logsum = thrust::transform_reduce(input_data+t*size, input_data+(t+1)*size, expm_functor(maxInput), (float)0, thrust::plus<float>());
    logsum = maxInput + log(logsum);
    thrust::transform(input_data+t*size, input_data+(t+1)*size, output_data+t*size, addvalue_functor(-logsum));
  }

  THCudaTensor_free(input);
  return 1;
}

static int cunn_TemporalLogSoftMax_backward(lua_State *L)
{
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, torch_CudaTensor_id);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", torch_CudaTensor_id);
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", torch_CudaTensor_id);
  long size = THCudaTensor_size(output, 1);

  output = THCudaTensor_newContiguous(output);
  gradOutput = THCudaTensor_newContiguous(gradOutput);
  THCudaTensor_resizeAs(gradInput, output);

  thrust::device_ptr<float> output_data(THCudaTensor_data(output));
  thrust::device_ptr<float> gradOutput_data(THCudaTensor_data(gradOutput));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(gradInput));

  for(long t = 0; t < output->size[0]; t++)
  {
    float sum = thrust::reduce(gradOutput_data+t*size, gradOutput_data+(t+1)*size, 0, thrust::plus<float>());
    thrust::transform(output_data+t*size, output_data+(t+1)*size, gradOutput_data+t*size, gradInput_data+t*size, logsoftmaxbackward_functor(sum));
  }

  THCudaTensor_free(gradOutput);
  THCudaTensor_free(output);
  return 1;
}

static const struct luaL_Reg cunn_TemporalLogSoftMax__ [] = {
  {"TemporalLogSoftMax_forward", cunn_TemporalLogSoftMax_forward},
  {"TemporalLogSoftMax_backward", cunn_TemporalLogSoftMax_backward},
  {NULL, NULL}
};

static void cunn_TemporalLogSoftMax_init(lua_State *L)
{
  luaT_pushmetaclass(L, torch_CudaTensor_id);
  luaT_registeratname(L, cunn_TemporalLogSoftMax__, "nn");
  lua_pop(L,1);
}
