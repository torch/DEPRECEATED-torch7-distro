
#ifndef DIVUP
#define DIVUP(x,y) (((x) + (y) - 1) / (y))
#endif

#define MIN(a,b) (a) < (b) ? (a) : (b)

#ifndef assert
#define assert(e)  \
    if (!(e)) { \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting..."); \
    };
#endif

#include "SpatialConvolutionCUDA/updateOutput.cu"
#include "SpatialConvolutionCUDA/updateGradInput.cu"
#include "SpatialConvolutionCUDA/accGradParameters.cu"

static int cunn_SpatialConvolutionCUDA_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  luaL_argcheck(L, input->nDimension == 4, 2, "4D (batch mode) tensor is expected");

  long nOutputPlane = weight->size[3];
  long nInputPlane  = weight->size[0];
  long kH           = weight->size[1];
  long kW           = weight->size[2];
  long inputHeight  = input->size[1];
  long inputWidth   = input->size[2];
  long batchSize    = input->size[3];
  long outputHeight = (padding + inputHeight - kH) / dH + 1;
  long outputWidth  = (padding + inputWidth - kW) / dW + 1;

  // resize output
  THCudaTensor_resize4d(output, nOutputPlane, outputHeight, outputWidth, batchSize);
  
  // asserts
  luaL_argcheck(L, inputWidth == inputHeight, 1, "input must be square");
  luaL_argcheck(L, kW == kW, 1, "kH must be equal to kW");
  luaL_argcheck(L, dH == dW, 1, "dH must be equal to dW");

  // all the data must be contiguous: 
  luaL_argcheck(L, THCudaTensor_isContiguous(input), 2, "input must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(weight), 1, "weight must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(output), 1, "output must be contiguous");

  // raw pointers 
  float *input_data = THCudaTensor_data(input);
  float *weight_data = THCudaTensor_data(weight);
  float *output_data = THCudaTensor_data(output);

  // convolutions
  spatialConv_updateOutput(
    input_data, weight_data, output_data,
    nInputPlane, inputHeight, inputWidth, batchSize,
    nOutputPlane, outputHeight, outputWidth,
    kH, kW,
    -floor((double)padding/2), dW,
    0, 1, true
  );
  
  return 1;
}

static int cunn_SpatialConvolutionCUDA_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  
  long nOutputPlane = weight->size[3];
  long nInputPlane  = weight->size[0];
  long kH           = weight->size[1];
  long kW           = weight->size[2];
  long inputHeight  = input->size[1];
  long inputWidth   = input->size[2];
  long batchSize    = input->size[3];
  long outputHeight = (padding + inputHeight - kH) / dH + 1;
  long outputWidth  = (padding + inputWidth - kW) / dW + 1;

  // resize gradInput
  THCudaTensor_resize4d(gradInput, nInputPlane, inputHeight, inputWidth, batchSize);
  
  // asserts
  luaL_argcheck(L, inputWidth == inputHeight, 1, "input must be square");
  luaL_argcheck(L, kW == kW, 1, "kH must be equal to kW");
  luaL_argcheck(L, dH == dW, 1, "dH must be equal to dW");

  // all the data must be contiguous: 
  luaL_argcheck(L, THCudaTensor_isContiguous(gradInput), 2, "input must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(weight), 1, "weight must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(gradOutput), 1, "output must be contiguous");

  // raw pointers 
  float *gradInput_data = THCudaTensor_data(gradInput);
  float *weight_data = THCudaTensor_data(weight);
  float *gradOutput_data = THCudaTensor_data(gradOutput);

  // convolutions
  spatialConv_updateGradInput(
    gradOutput_data, weight_data, gradInput_data, 
    nInputPlane, inputHeight, inputWidth, batchSize,
    nOutputPlane, outputHeight, outputWidth,
    kH, kW,
    -floor((double)padding/2), dW,
    0, 1, true
  );

  return 1;
}

static int cunn_SpatialConvolutionCUDA_accGradParameters(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padding = luaT_getfieldcheckint(L, 1, "padding");
  int partialSum = luaT_getfieldcheckint(L, 1, "partialSum");
  float scale = luaL_optnumber(L, 4, 1);

  long nOutputPlane = gradWeight->size[3];
  long nInputPlane  = gradWeight->size[0];
  long kH           = gradWeight->size[1];
  long kW           = gradWeight->size[2];
  long inputHeight  = input->size[1];
  long inputWidth   = input->size[2];
  long batchSize    = input->size[3];
  long outputHeight = (padding + inputHeight - kH) / dH + 1;
  long outputWidth  = (padding + inputWidth - kW) / dW + 1;
  
  // asserts
  luaL_argcheck(L, inputWidth == inputHeight, 1, "input must be square");
  luaL_argcheck(L, kW == kW, 1, "kH must be equal to kW");
  luaL_argcheck(L, dH == dW, 1, "dH must be equal to dW");

  if (partialSum) {
    // compute partial gradients for outputHeight*outputWidth/partialSum groups of filters separately
    gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeightPartial", "torch.CudaTensor");
    THCudaTensor_resize4d(gradWeight, outputHeight*outputWidth/partialSum, nInputPlane, kH*kW, nOutputPlane);
    // numModuleY*numModulesX/partialSum, numFilterColors, filterPixels, numFilters
  }

  // all the data must be contiguous: 
  luaL_argcheck(L, THCudaTensor_isContiguous(input), 2, "input must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(gradWeight), 1, "weight must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(gradOutput), 1, "output must be contiguous");

  // raw pointers 
  float *input_data = THCudaTensor_data(input);
  float *gradWeight_data = THCudaTensor_data(gradWeight);
  float *gradOutput_data = THCudaTensor_data(gradOutput);

  // convolutions
  spatialConv_accGradParameters(
    input_data, gradOutput_data, gradWeight_data,
    nInputPlane, inputHeight, inputWidth, batchSize,
    nOutputPlane, outputHeight, outputWidth,
    kH, kW,
    -floor((double)padding/2), dW,
    0, scale, partialSum
  );

  return 0;
}

static const struct luaL_Reg cunn_SpatialConvolutionCUDA__ [] = {
  {"SpatialConvolutionCUDA_updateOutput", cunn_SpatialConvolutionCUDA_updateOutput},
  {"SpatialConvolutionCUDA_updateGradInput", cunn_SpatialConvolutionCUDA_updateGradInput},
  {"SpatialConvolutionCUDA_accGradParameters", cunn_SpatialConvolutionCUDA_accGradParameters},
  {NULL, NULL}
};

static void cunn_SpatialConvolutionCUDA_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialConvolutionCUDA__, "nn");
  lua_pop(L,1);
}
