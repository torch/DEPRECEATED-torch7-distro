#define MINUS_LOG_THRESHOLD -18.42

struct logaddforward_functor
{
  __host__ __device__ float operator()(const float& log_a_, const float& log_b_) const
  {
    float minusdif;
    float log_a = log_a_;
    float log_b = log_b_;

    if (log_a < log_b)
    {
      log_a = log_b_;
      log_b = log_a_;
    }

    minusdif = log_b - log_a;

    if (minusdif < MINUS_LOG_THRESHOLD)
      return log_a;
    else
      return log_a + log1p(exp(minusdif));
  }
};

struct addvalue_functor
{
  const float value;

  addvalue_functor(float value_) : value(value_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return (x+value);
  }
};

static int cunn_LogSoftMax_forward(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, torch_CudaTensor_id);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", torch_CudaTensor_id);
  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);

  THCudaTensor_resizeAs(output, input);
  THCudaTensor_copy(output, input);

  thrust::device_ptr<float> output_data(THCudaTensor_data(output));
  thrust::device_ptr<float> input_data(THCudaTensor_data(input));

  float logsum = thrust::reduce(input_data, input_data+size, (float)(-THInf), logaddforward_functor());
  thrust::transform(input_data, input_data+size, output_data, addvalue_functor(-logsum));

  THCudaTensor_free(input);
  return 1;
}

struct logsoftmaxbackward_functor
{
  float value;

  logsoftmaxbackward_functor(float value_) : value(value_) {}

  __host__ __device__ float operator()(const float& output, const float& gradOutput) const
  {
    return gradOutput - exp(output)*value;
  }
};

static int cunn_LogSoftMax_backward(lua_State *L)
{
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, torch_CudaTensor_id);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", torch_CudaTensor_id);
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", torch_CudaTensor_id);
  long size = THCudaTensor_nElement(output);

  output = THCudaTensor_newContiguous(output);
  gradOutput = THCudaTensor_newContiguous(gradOutput);

  THCudaTensor_resizeAs(gradInput, output);

  thrust::device_ptr<float> output_data(THCudaTensor_data(output));
  thrust::device_ptr<float> gradOutput_data(THCudaTensor_data(gradOutput));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(gradInput));

  float sum = thrust::reduce(gradOutput_data, gradOutput_data+size, 0, thrust::plus<float>());

  thrust::transform(output_data, output_data+size, gradOutput_data, gradInput_data, logsoftmaxbackward_functor(sum));

  THCudaTensor_free(gradOutput);
  THCudaTensor_free(output);
  return 1;
}

static const struct luaL_Reg cunn_LogSoftMax__ [] = {
  {"LogSoftMax_forward", cunn_LogSoftMax_forward},
  {"LogSoftMax_backward", cunn_LogSoftMax_backward},
  {NULL, NULL}
};

static void cunn_LogSoftMax_init(lua_State *L)
{
  luaT_pushmetaclass(L, torch_CudaTensor_id);
  luaT_registeratname(L, cunn_LogSoftMax__, "nn");
  lua_pop(L,1);
}
