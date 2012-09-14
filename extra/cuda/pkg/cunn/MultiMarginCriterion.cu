#define MULTIMARGIN_THREADS 128

__global__ void cunn_MultiMarginCriterion_updateOutput_kernel(float *output, float *input, float *target, int nframe, int dim, int sizeaverage)
{
  __shared__ float buffer[MULTIMARGIN_THREADS];
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *output_k = output + k;
  int target_k = ((int)target[k])-1;
  float input_target_k = input_k[target_k];

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for(int i = i_start; i < i_end; i += i_step)
  {
    float z = 1 - input_target_k + input_k[i];
    if(i == target_k)
      continue;
    
    if(z > 0)
      buffer[threadIdx.x] += z;
  }
  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float sum = 0;
    for (int i=0; i<blockDim.x; i++)
      sum += buffer[i];

    if(sizeaverage)
      *output_k = sum/dim;
    else
      *output_k = sum;
  }
}


__global__ void cunn_MultiMarginCriterion_updateGradInput_kernel(float *gradInput, float *input, float *target, int nframe, int dim, int sizeaverage)
{
  __shared__ float buffer[MULTIMARGIN_THREADS];
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *gradInput_k = gradInput + k*dim;
  int target_k = ((int)target[k])-1;
  float input_target_k = input_k[target_k];
  float g = (sizeaverage ? 1./((float)dim) : 1.);

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float z = 1 - input_target_k + input_k[i];
    if(i == target_k)
      continue;
    
    if(z > 0)
    {
      buffer[threadIdx.x] -= g;
      gradInput_k[i] = g;
    }
    else
      gradInput_k[i] = 0;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float gradInput_target_k = 0;
    for (int i=0; i<blockDim.x; i++)
      gradInput_target_k += buffer[i];
    gradInput_k[target_k] = gradInput_target_k;
  }
}

static int cunn_MultiMarginCriterion_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  int sizeaverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");

  input = THCudaTensor_newContiguous(input);

  if(input->nDimension == 1)
  {
    float target_ = luaL_checknumber(L, 3);
    THCudaStorage *target = THCudaStorage_newWithSize(1);
    THCudaStorage *output = THCudaStorage_newWithSize(1);
    dim3 blocks(1);
    dim3 threads(MULTIMARGIN_THREADS);

    THCudaStorage_fill(target, target_);

    cunn_MultiMarginCriterion_updateOutput_kernel<<<blocks,threads>>>(output->data,
                                                                 THCudaTensor_data(input),
                                                                 target->data,
                                                                 1, input->size[0],
                                                                 sizeaverage);
    lua_pushnumber(L, THCudaStorage_get(output, 0));

    THCudaStorage_free(output);
    THCudaStorage_free(target);
  }
  else if(input->nDimension == 2)
  {
    THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *output = THCudaTensor_newWithSize1d(input->size[0]);
    dim3 blocks(input->size[0]);
    dim3 threads(MULTIMARGIN_THREADS);
    cunn_MultiMarginCriterion_updateOutput_kernel<<<blocks,threads>>>(THCudaTensor_data(output),
                                                                 THCudaTensor_data(input),
                                                                 THCudaTensor_data(target),
                                                                 input->size[0], input->size[1],
                                                                 sizeaverage);
    lua_pushnumber(L, THCudaTensor_sumall(output));
    THCudaTensor_free(output);
  }
  else
    THError("vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  lua_pushstring(L, "output");
  lua_pushvalue(L, -2);
  lua_rawset(L, 1);

  THCudaTensor_free(input);
  return 1;
}

static int cunn_MultiMarginCriterion_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  int sizeaverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");

  THCudaTensor_resizeAs(gradInput, input);

  if(gradInput->nDimension == 1)
  {
    float target_ = luaL_checknumber(L, 3);
    THCudaTensor *target = THCudaTensor_newWithSize1d(1);
    dim3 blocks(1);
    dim3 threads(LOGSOFTMAX_THREADS);

    THCudaTensor_fill(target, target_);

    cunn_MultiMarginCriterion_updateGradInput_kernel<<<blocks,threads>>>(THCudaTensor_data(gradInput),
                                                                  THCudaTensor_data(input),
                                                                  THCudaTensor_data(target),
                                                                  1, gradInput->size[0],
                                                                  sizeaverage);

    THCudaTensor_free(target);
  }
  else if(gradInput->nDimension == 2)
  {
    THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    dim3 blocks(gradInput->size[0]);
    dim3 threads(LOGSOFTMAX_THREADS);

    cunn_MultiMarginCriterion_updateGradInput_kernel<<<blocks,threads>>>(THCudaTensor_data(gradInput),
                                                                  THCudaTensor_data(input),
                                                                  THCudaTensor_data(target),
                                                                  gradInput->size[0], gradInput->size[1],
                                                                  sizeaverage);
  }
  else
    THError("vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  return 1;
}

static const struct luaL_Reg cunn_MultiMarginCriterion__ [] = {
  {"MultiMarginCriterion_updateOutput", cunn_MultiMarginCriterion_updateOutput},
  {"MultiMarginCriterion_updateGradInput", cunn_MultiMarginCriterion_updateGradInput},
  {NULL, NULL}
};

static void cunn_MultiMarginCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_MultiMarginCriterion__, "nn");
  lua_pop(L,1);
}
