
static int cunn_SpatialConvolution_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  int dimw = 2;
  int dimh = 1;
  if (input->nDimension == 4)
  {
    dimw++;
    dimh++;
  }

  long nOutputPlane = weight->size[0];
  long kW           = weight->size[3];
  long kH           = weight->size[2];
  long inputWidth   = input->size[dimw];
  long inputHeight  = input->size[dimh];
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;

  if (input->nDimension == 3)
  {
    THCudaTensor_resize3d(output, nOutputPlane, outputHeight, outputWidth);

    /* add bias first */
    long k;
    THCudaTensor *outputPlane = THCudaTensor_new();
    for(k=0; k<nOutputPlane; k++) {
      THCudaTensor_select(outputPlane, output, 0, k);
      THCudaTensor_fill(outputPlane, THCudaTensor_get1d(bias, k));
    }
    THCudaTensor_free(outputPlane);

    /* do convolutions */
    THCudaTensor_conv2Dmv(output, 1.0, input, weight, dH, dW, "vx");
  }
  else
  {
    THCudaTensor_resize4d(output, input->size[0],nOutputPlane, outputHeight, outputWidth);

    /* add bias first */
    long k,p;
    THCudaTensor *outputPlane = THCudaTensor_new();
    THCudaTensor *outputBatch = THCudaTensor_new();
    for(p=0; p<input->size[0]; p++) {
      THCudaTensor_select(outputBatch, output, 0, p);
      for(k=0; k<nOutputPlane; k++) {
	THCudaTensor_select(outputPlane, outputBatch, 0, k);
	THCudaTensor_fill(outputPlane, THCudaTensor_get1d(bias, k));
      }
    }
    THCudaTensor_free(outputPlane);
    THCudaTensor_free(outputBatch);

    /* do convolutions */
    THCudaTensor_conv2Dmm(output, 1.0, input, weight, dH, dW, "vx");
  }
  return 1;
}

static int cunn_SpatialConvolution_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  luaL_argcheck(L, dW == 1, 1, "dW must be 1 (this is only a limit for CudaTensors)");
  luaL_argcheck(L, dH == 1, 1, "dH must be 1 (this is only a limit for CudaTensors)");

  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  if (input->nDimension == 3)
  {
    /* check dims */
    THArgCheck(nOutputPlane == gradOutput->size[0], 1, "Number of output features is not equal to nOutputPlane");

    /* gradient to input */
    THCudaTensor *tweight = THCudaTensor_newTranspose(weight,0,1);
    THCudaTensor_conv2Dmv(gradInput, 0.0, gradOutput, tweight, dH, dW, "fc");
    THCudaTensor_free(tweight);
  }
  else 
  {
    /* check dims */
    THArgCheck(nOutputPlane == gradOutput->size[1], 1, "Number of output features is not equal to nOutputPlane");

    /* gradient to input */
    THCudaTensor *tweight = THCudaTensor_newTranspose(weight,0,1);
    THCudaTensor_conv2Dmm(gradInput, 0.0, gradOutput, tweight, dH, dW, "fc");
    THCudaTensor_free(tweight);    
  }

  return 1;
}

__global__ void compute_gradBias(float *gradBias, float *gradOutput, float scale,
                                 int output_n, int output_h, int output_w)
{
  // each block does a plane
  int k = blockIdx.x;
  float *gradOutput_k = gradOutput + (k + threadIdx.y*output_n)*output_h*output_w;

  // offsets
  int i_start = threadIdx.x;
  int i_end = output_w*output_h;
  int i_step = blockDim.x;

  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int nthreads = blockDim.x * blockDim.y;

  // sum output plane k into partial sum array
  __shared__ float sums[512];
  sums[tid] = 0;
  for (int i=i_start; i<i_end; i+=i_step) {
    sums[tid] += gradOutput_k[i];
  }
  __syncthreads();

  // reduce
  if (tid == 0) {
    for (int i=0; i<nthreads; i++)
      gradBias[k] += scale*sums[i];
  }
}

static int cunn_SpatialConvolution_accGradParameters(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  float scale = luaL_optnumber(L, 4, 1);

  luaL_argcheck(L, dW == 1, 1, "dW must be 1 (this will be fixed soon)");
  luaL_argcheck(L, dH == 1, 1, "dH must be 1 (this will be fixed soon)");

  THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");

  float *gradBias_data = THCudaTensor_data(gradBias);
  float *gradOutput_data = THCudaTensor_data(gradOutput);

  if (input->nDimension == 3)
  {
    /* check dims */
    THArgCheck(nOutputPlane == gradOutput->size[0], 1, "Number of output features is not equal to nOutputPlane");

    /* gradient to bias */
    dim3 blocks(nOutputPlane);
    dim3 threads(32);
    compute_gradBias <<<blocks, threads>>> (gradBias_data, gradOutput_data, scale,
                                            gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);

    /* gradient to kernels */
    THCudaTensor_conv2DRevger(gradWeight, 1.0, scale, input, gradOutput, dH, dW);
  }
  else
  {
    /* check dims */
    THArgCheck(nOutputPlane == gradOutput->size[1], 1, "Number of output features is not equal to nOutputPlane");

    /* gradient to bias */
    dim3 blocks(nOutputPlane);
    long sl;
    for (sl=0; sl<gradOutput->size[0]; sl+=16) {
      int cst = 16;
      if ((cst+sl) > gradOutput->size[0]) cst = gradOutput->size[0] - sl;
      dim3 threads(16, cst);
      compute_gradBias <<<blocks, threads>>> (gradBias_data, gradOutput_data + sl*gradOutput->stride[0], scale,
                                              gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
    }

    /* gradient to kernels */
    THCudaTensor_conv2DRevgerm(gradWeight, 1.0, scale, input, gradOutput, dH, dW);
  }

  return 0;
}

static const struct luaL_Reg cunn_SpatialConvolution__ [] = {
  {"SpatialConvolution_updateOutput", cunn_SpatialConvolution_updateOutput},
  {"SpatialConvolution_updateGradInput", cunn_SpatialConvolution_updateGradInput},
  {"SpatialConvolution_accGradParameters", cunn_SpatialConvolution_accGradParameters},
  {NULL, NULL}
};

static void cunn_SpatialConvolution_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialConvolution__, "nn");
  lua_pop(L,1);
}
