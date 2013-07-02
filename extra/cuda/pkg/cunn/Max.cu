
/*
 * Description:
 *    this function finds the max along the innermost dimension
 *    Nd input, (N-1)d output, (N-1)d argmax
 */
__global__ void max_output(float *input, float *output, float *indices,
                           long nrows, long ncols)
{
  // output offset:
  long o = threadIdx.x + blockDim.x * blockIdx.x;
  if (o >= nrows) return;

  // input offset:
  long i = o * ncols;

  // move pointers
  input = input + i;

  // compute max:
  float max = input[0];
  long argmax = 0;
  long ii;
  for (ii=1; ii<ncols; ii++) {
      float val = input[ii];
      if (val > max) {
          max = val;
          argmax = ii;
      }
  }

  // store
  output[o] = max;
  indices[o] = argmax+1;
}

__global__ void max_gradInput(float *input, float *output, float *indices,
                              long nrows, long ncols)
{
  // output offset:
  long o = threadIdx.x + blockDim.x * blockIdx.x;
  if (o >= nrows) return;

  // input offset:
  long i = o * ncols;

  // bprop max gradient:
  long idx = indices[o]-1;
  input[i+idx] = output[o];
}

static int cunn_Max_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  int dimension = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");
  luaL_argcheck(L, dimension == input->nDimension-1, 2, "only supported dimension is innermost (CUDA kernel only)");

  input = THCudaTensor_newContiguous(input);

  THLongStorage *dim = THLongStorage_newWithSize(input->nDimension);
  long i;
  for(i = 0; i < input->nDimension; i++)
    dim->data[i] = input->size[i];
  dim->data[dimension] = 1;
  THCudaTensor_resize(output, dim, NULL);
  THCudaTensor_resize(indices, dim, NULL);
  THLongStorage_free(dim);

  float *input_data = THCudaTensor_data(input);
  float *output_data = THCudaTensor_data(output);
  float *indices_data = THCudaTensor_data(indices);

  long nrows = THCudaTensor_nElement(output);
  long ncols = input->size[dimension];

  // cuda blocks & threads:
  long nthreads = 256;
  long nblocks = ceil((float)nrows / nthreads);
  dim3 blocks(nblocks);
  dim3 threads(nthreads);

  // kernel:
  max_output <<<blocks, threads>>> (input_data, output_data, indices_data, nrows, ncols);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in Max.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
 
  // final cut:
  THCudaTensor_free(input); 
  THCudaTensor_select(output, NULL, dimension, 0);

  return 1;
}

static int cunn_Max_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");
  int dimension  = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THCudaTensor *gradInput  = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THCudaTensor_resizeAs(gradInput, input);
  THCudaTensor_zero(gradInput);
  
  float *gradInput_data = THCudaTensor_data(gradInput);
  float *gradOutput_data = THCudaTensor_data(gradOutput);
  float *indices_data = THCudaTensor_data(indices);
  
  long nrows = THCudaTensor_nElement(gradOutput);
  long ncols = gradInput->size[dimension];

  // cuda blocks & threads:
  long nthreads = 256;
  long nblocks = ceil((float)nrows / nthreads);
  dim3 blocks(nblocks);
  dim3 threads(nthreads);
  
  // kernel:
  max_gradInput <<<blocks, threads>>> (gradInput_data, gradOutput_data, indices_data, nrows, ncols);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in Max.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  return 1;
}

static const struct luaL_Reg cunn_Max__ [] = {
  {"Max_updateOutput", cunn_Max_updateOutput},
  {"Max_updateGradInput", cunn_Max_updateGradInput},
  {NULL, NULL}
};

static void cunn_Max_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Max__, "nn");
  lua_pop(L,1);
}
