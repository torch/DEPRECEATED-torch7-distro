#define MINUS_LOG_THRESHOLD -18.42
#define LOGSOFTMAX_THREADS 128

struct addvalue_functor
{
  const float value;

  addvalue_functor(float value_) : value(value_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return (x+value);
  }
};

__global__ void cunn_LogSoftMax_updateOutput_kernel(float *output, float *input, int nframe, int dim)
{
  __shared__ float buffer[LOGSOFTMAX_THREADS+1];
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *output_k = output + k*dim;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // max?
  buffer[threadIdx.x] = -THInf;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float z = input_k[i];
    if(buffer[threadIdx.x] < z)
      buffer[threadIdx.x] = z;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float max_k = -THInf;
    for (int i=0; i<blockDim.x; i++)
    {
      if(max_k < buffer[i])
        max_k = buffer[i];
    }
    buffer[LOGSOFTMAX_THREADS] = max_k;
  }

  __syncthreads();

  // logadd?
  float max_k = buffer[LOGSOFTMAX_THREADS];
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
    buffer[threadIdx.x] += __expf(input_k[i]-max_k);

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float logsum_k = 0;
    for (int i=0; i<blockDim.x; i++)
      logsum_k += buffer[i];
    buffer[LOGSOFTMAX_THREADS] = max_k + __logf(logsum_k);
  }

  __syncthreads();

  // logsoftmax
  float logsum_k = buffer[LOGSOFTMAX_THREADS];
  for (int i=i_start; i<i_end; i+=i_step)
    output_k[i] = input_k[i] - logsum_k;
}


__global__ void cunn_LogSoftMax_updateGradInput_kernel(float *gradInput, float *output, float *gradOutput, int nframe, int dim)
{
  __shared__ float buffer[LOGSOFTMAX_THREADS];
  int k = blockIdx.x;
  float *gradInput_k = gradInput + k*dim;
  float *output_k = output + k*dim;
  float *gradOutput_k = gradOutput + k*dim;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // sum?
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
    buffer[threadIdx.x] += gradOutput_k[i];

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float sum_k = 0;
    for (int i=0; i<blockDim.x; i++)
      sum_k += buffer[i];
    buffer[0] = sum_k;
  }

  __syncthreads();

  float sum_k = buffer[0];
  for (int i=i_start; i<i_end; i+=i_step)
    gradInput_k[i] = gradOutput_k[i] - __expf(output_k[i])*sum_k;
}

static int cunn_LogSoftMax_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  input = THCudaTensor_newContiguous(input);
  THCudaTensor_resizeAs(output, input);

  if(input->nDimension == 1)
  {
    dim3 blocks(1);
    dim3 threads(LOGSOFTMAX_THREADS);
    cunn_LogSoftMax_updateOutput_kernel<<<blocks,threads>>>(THCudaTensor_data(output), THCudaTensor_data(input), 1, input->size[0]);
  }
  else if(input->nDimension == 2)
  {
    dim3 blocks(input->size[0]);
    dim3 threads(LOGSOFTMAX_THREADS);
    cunn_LogSoftMax_updateOutput_kernel<<<blocks,threads>>>(THCudaTensor_data(output), THCudaTensor_data(input), input->size[0], input->size[1]);
  }
  else
    THError("vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(input);
  return 1;
}

struct logsoftmaxupdateGradInput_functor
{
  float value;

  logsoftmaxupdateGradInput_functor(float value_) : value(value_) {}

  __host__ __device__ float operator()(const float& output, const float& gradOutput) const
  {
    return gradOutput - exp(output)*value;
  }
};

static int cunn_LogSoftMax_updateGradInput(lua_State *L)
{
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  output = THCudaTensor_newContiguous(output);
  gradOutput = THCudaTensor_newContiguous(gradOutput);

  THCudaTensor_resizeAs(gradInput, output);

  if(gradInput->nDimension == 1)
  {
    dim3 blocks(1);
    dim3 threads(LOGSOFTMAX_THREADS);

    cunn_LogSoftMax_updateGradInput_kernel<<<blocks,threads>>>(THCudaTensor_data(gradInput),
                                                        THCudaTensor_data(output),
                                                        THCudaTensor_data(gradOutput),
                                                        1, gradInput->size[0]);
  }
  else if(gradInput->nDimension == 2)
  {
    dim3 blocks(gradInput->size[0]);
    dim3 threads(LOGSOFTMAX_THREADS);

    cunn_LogSoftMax_updateGradInput_kernel<<<blocks,threads>>>(THCudaTensor_data(gradInput),
                                                        THCudaTensor_data(output),
                                                        THCudaTensor_data(gradOutput),
                                                        gradInput->size[0], gradInput->size[1]);
  }
  else
    THError("vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(gradOutput);
  THCudaTensor_free(output);
  return 1;
}

static const struct luaL_Reg cunn_LogSoftMax__ [] = {
  {"LogSoftMax_updateOutput", cunn_LogSoftMax_updateOutput},
  {"LogSoftMax_updateGradInput", cunn_LogSoftMax_updateGradInput},
  {NULL, NULL}
};

static void cunn_LogSoftMax_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_LogSoftMax__, "nn");
  lua_pop(L,1);
}
