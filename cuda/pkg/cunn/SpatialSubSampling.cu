
#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

/*
 * Description:
 *    this function subsamples an input 3D tensor along dimensions 1 and 2
 *    3D input, 3D output, 1D weight, 1D bias
 */
__global__ void subsample(float *input, float *output, float *weight, float *bias,
                          int input_n, int input_h, int input_w,
                          int kH, int kW, int dH, int dW,
                          int patch_h, int patch_w)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int k = blockIdx.x;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = threadIdx.y * patch_h;
  int yy_end = yy_start + patch_h; if (yy_end > output_h) yy_end = output_h;
  int yy_step = 1;

  // select input/output plane
  output = output + k*output_w*output_h;
  input = input + k*input_w*input_h;

  // Get the good mask for (k,i) (k out, i in)
  float the_weight = weight[k];

  // Initialize to the bias
  float the_bias = bias[k];

  // For all output pixels...
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      // Compute the mean of the input image...
      float *ptr_input = input + yy*dH*input_w + xx*dW;
      float *ptr_output = output + yy*output_w + xx;
      float sum = 0;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++)
          sum += ptr_input[kx];
        ptr_input += input_w; // next input line
      }
      // Update output
      *ptr_output = the_weight*sum + the_bias;
    }
  }
}

/*
 * Description:
 *    this function computes the gradWeight from input and gradOutput
 */
__global__ void subgradweight(float *input, float *gradOutput, float *gradWeight, float *gradBias,
                              int input_n, int input_h, int input_w,
                              int kH, int kW, int dH, int dW,
                              int patch_h, int patch_w)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int k = blockIdx.x;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = threadIdx.y * patch_h;
  int yy_end = yy_start + patch_h; if (yy_end > output_h) yy_end = output_h;
  int yy_step = 1;

  // select input/output plane
  gradOutput = gradOutput + k*output_w*output_h;
  input = input + k*input_w*input_h;

  // create array to hold partial sums
  __shared__ float sums[CUDA_MAX_THREADS];
  float *psum = &sums[blockDim.x*threadIdx.y + threadIdx.x];
  *psum = 0;

  // compute partial sums
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      float *ptr_input = input + yy*dH*input_w + xx*dW;
      float *ptr_gradOutput = gradOutput + yy*output_w + xx;
      float z = *ptr_gradOutput;
      long kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++) {
          *psum += z * ptr_input[kx];
        }
        ptr_input += input_w;
      }
    }
  }
  __syncthreads();

  // reduce: accumulate all partial sums to produce final gradWeight
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for(int i = 0; i < blockDim.x*blockDim.y; i++) gradWeight[k] += sums[i];
  }

  // compute gradBias
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) { 
    for(int i = 0; i < output_h*output_w; i++) gradBias[k] += gradOutput[i];
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 */
__global__ void subgradinput(float *gradInput, float *gradOutput, float *weight,
                             int input_n, int input_h, int input_w,
                             int kH, int kW, int dH, int dW,
                             int patch_h, int patch_w)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int k = blockIdx.x;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = threadIdx.y * patch_h;
  int yy_end = yy_start + patch_h; if (yy_end > output_h) yy_end = output_h;
  int yy_step = 1;

  // select input/output plane
  gradOutput = gradOutput + k*output_w*output_h;
  gradInput = gradInput + k*input_w*input_h;

  // get weight
  float the_weight = weight[k];

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
      float *ptr_gradOutput = gradOutput + yy*output_w + xx;
      float z = *ptr_gradOutput * the_weight;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++)
          ptr_gradInput[kx] += z;
        ptr_gradInput += input_w;
      }
    }
  }
}

static int cunn_SpatialSubSampling_forward(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, torch_CudaTensor_id);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", torch_CudaTensor_id);
  THCudaTensor *bias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "bias", torch_CudaTensor_id);
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", torch_CudaTensor_id);

  float *weight_data = THCudaTensor_data(weight);
  float *bias_data = THCudaTensor_data(bias);
  float *output_data;
  float *input_data;

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");

  long nInputCols = input->size[2];
  long nInputRows = input->size[1];
  long nOutputCols = (nInputCols - kW) / dW + 1;
  long nOutputRows = (nInputRows - kH) / dH + 1;

  luaL_argcheck(L, input->size[0] == nInputPlane, 2, "invalid number of input planes");
  luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

  input = THCudaTensor_newContiguous(input);
  input_data = THCudaTensor_data(input);

  THCudaTensor_resize3d(output, nInputPlane, nOutputRows, nOutputCols);
  output_data = THCudaTensor_data(output);

  // cuda blocks & threads:
  // we create one block per input map, and then try to have 256 threads per map
  // arranged in a 16x16 grid.
  int patch_w = (int)(pow(2, ceil(log2((float)nOutputCols))) / 16);
  if (patch_w < 2) patch_w = 2;
  int patch_h = (int)(pow(2, ceil(log2((float)nOutputRows))) / 16);
  if (patch_h < 2) patch_h = 2;
  dim3 blocks(nInputPlane);
  dim3 threads(nOutputCols / patch_w, nOutputRows / patch_h);
  if ((nOutputRows % patch_h) != 0) threads.y++;
  if ((nOutputCols % patch_w) != 0) threads.x++;

  // sync
  cudaDeviceSynchronize();

  // run subsample kernel
  subsample <<<blocks, threads>>> (input_data, output_data, weight_data, bias_data,
                                   nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW,
                                   patch_h, patch_w);

  // sync & clean
  cudaDeviceSynchronize();
  THCudaTensor_free(input);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in conv2Dmv: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static int cunn_SpatialSubSampling_backward(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, torch_CudaTensor_id);
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, torch_CudaTensor_id);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  luaL_argcheck(L, dW == kW, 1, "dW and kW must be equal (this will be fixed soon)");
  luaL_argcheck(L, dH == kH, 1, "dH and kH must be equal (this will be fixed soon)");

  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", torch_CudaTensor_id);
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", torch_CudaTensor_id);

  long nInputCols = input->size[2];
  long nInputRows = input->size[1];
  long nOutputCols = (nInputCols - kW) / dW + 1;
  long nOutputRows = (nInputRows - kH) / dH + 1;

  float *weight_data = THCudaTensor_data(weight);
  float *gradOutput_data = THCudaTensor_data(gradOutput);
  float *gradInput_data;

  THCudaTensor_resizeAs(gradInput, input);
  THCudaTensor_zero(gradInput);
  gradInput_data = THCudaTensor_data(gradInput);

  // cuda blocks & threads:
  // we create one block per input map, and then try to have 256 threads per map
  // arranged in a 16x16 grid.
  int patch_w = (int)(pow(2, ceil(log2((float)nOutputCols))) / 16);
  if (patch_w < 2) patch_w = 2;
  int patch_h = (int)(pow(2, ceil(log2((float)nOutputRows))) / 16);
  if (patch_h < 2) patch_h = 2;
  dim3 blocks(nInputPlane);
  dim3 threads(nOutputCols / patch_w, nOutputRows / patch_h);
  if ((nOutputRows % patch_h) != 0) threads.y++;
  if ((nOutputCols % patch_w) != 0) threads.x++;

  // sync
  cudaDeviceSynchronize();

  // run backward kernel
  subgradinput <<<blocks, threads>>> (gradInput_data, gradOutput_data, weight_data,
                                      nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW,
                                      patch_h, patch_w);

  // sync & clean
  cudaDeviceSynchronize();

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in conv2Dmv: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static int cunn_SpatialSubSampling_accGradParameters(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, torch_CudaTensor_id);
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, torch_CudaTensor_id);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  luaL_argcheck(L, dW == kW, 1, "dW and kW must be equal (this will be fixed soon)");
  luaL_argcheck(L, dH == kH, 1, "dH and kH must be equal (this will be fixed soon)");

  THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", torch_CudaTensor_id);
  THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", torch_CudaTensor_id);

  long nInputCols = input->size[2];
  long nInputRows = input->size[1];
  long nOutputCols = (nInputCols - kW) / dW + 1;
  long nOutputRows = (nInputRows - kH) / dH + 1;

  float *gradWeight_data = THCudaTensor_data(gradWeight);
  float *gradBias_data = THCudaTensor_data(gradBias);
  float *gradOutput_data = THCudaTensor_data(gradOutput);
  float *input_data;

  input = THCudaTensor_newContiguous(input);
  input_data = THCudaTensor_data(input);

  // cuda blocks & threads:
  // we create one block per input map, and then try to have 256 threads per map
  // arranged in a 16x16 grid.
  int patch_w = (int)(pow(2, ceil(log2((float)nOutputCols))) / 16);
  if (patch_w < 2) patch_w = 2;
  else if (patch_w > 32) patch_w = 32;
  int patch_h = (int)(pow(2, ceil(log2((float)nOutputRows))) / 16);
  if (patch_h < 2) patch_h = 2;
  else if (patch_h > 32) patch_h = 32;
  dim3 blocks(nInputPlane);
  dim3 threads(nOutputCols / patch_w, nOutputRows / patch_h);
  if ((nOutputRows % patch_h) != 0) threads.y++;
  if ((nOutputCols % patch_w) != 0) threads.x++;

  // sync
  cudaDeviceSynchronize();

  // run gradweight kernel
  subgradweight <<<blocks, threads>>> (input_data, gradOutput_data, gradWeight_data, gradBias_data,
                                       nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW,
                                       patch_h, patch_w);

  // sync & clean
  cudaDeviceSynchronize();
  THCudaTensor_free(input);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in conv2Dmv: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 0;
}

static const struct luaL_Reg cunn_SpatialSubSampling__ [] = {
  {"SpatialSubSampling_forward", cunn_SpatialSubSampling_forward},
  {"SpatialSubSampling_backward", cunn_SpatialSubSampling_backward},
  {"SpatialSubSampling_accGradParameters", cunn_SpatialSubSampling_accGradParameters},
  {NULL, NULL}
};

static void cunn_SpatialSubSampling_init(lua_State *L)
{
  luaT_pushmetaclass(L, torch_CudaTensor_id);
  luaT_registeratname(L, cunn_SpatialSubSampling__, "nn");
  lua_pop(L,1);
}
