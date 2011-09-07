#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionOmp.c"
#else

static int nnOmp_(SpatialConvolution_forwardOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  setompnthread(L,1,"nThread");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");

  long nOutputPlane = weight->size[0];
  long nInputPlane  = weight->size[1];
  long kW           = weight->size[3];
  long kH           = weight->size[2];
  long inputWidth   = input->size[2];
  long inputHeight  = input->size[1];
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;

  THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);


  /* add bias */
  long i;
  /*THTensor *outn = THTensor_(new)();*/
  real* bias_data = THTensor_(data)(bias);
  real* output_data = THTensor_(data)(output);
#pragma omp parallel for private(i)
  for (i=0; i<bias->size[0]; i++)
  {
    /*THTensor_(select)(outn,output,0,i);*/
    /*TH_TENSOR_APPLY(real,outn, *outn_data = bias_data[i];);*/
    real *ptr_output = output_data + i*outputWidth*outputHeight;
    long j;
    for(j = 0; j < outputWidth*outputHeight; j++)
      ptr_output[j] = bias_data[i];
  }
  /*THTensor_(free)(outn);*/

  /* do convolutions */
  THOmpLab_(conv2Dmv)(output, 1.0, 1.0, input, weight, dH, dW, "vx");

  return 1;
}


static int nnOmp_(SpatialConvolution_backwardOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  setompnthread(L,1,"nThread");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THArgCheck( nOutputPlane == gradOutput->size[0], 1, "Number of output features is not equal to nOutputPlane" );

  long k;

  real *gradOutput_data = THTensor_(data)(gradOutput);

  /* gradient to input */
  THTensor *tweight = THTensor_(newTranspose)(weight,0,1);
  THOmpLab_(conv2Dmv)(gradInput, 0.0, 1.0, gradOutput, tweight, dH, dW, "fc");
  THTensor_(free)(tweight);

  return 1;
}

static int nnOmp_(SpatialConvolution_accGradParametersOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  real scale = luaL_optnumber(L, 4, 1);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  setompnthread(L,1,"nThread");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_(Tensor_id));

  THArgCheck( nOutputPlane == gradOutput->size[0], 1, "Number of output features is not equal to nOutputPlane" );

  long k;

  /* gradient to bias */
  real *gradBias_data = THTensor_(data)(gradBias);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  long noutSlice = gradOutput->size[1]*gradOutput->size[2];
  /*THTensor* gradOutSlice = THTensor_(new)();*/


#pragma omp parallel for private(k)
  for(k = 0; k < nOutputPlane; k++)
  {
    /*THTensor_(select)(gradOutSlice, gradOutput, 0, k);*/
    real *ptr_gradOutput = gradOutput_data + k*noutSlice;
    long l;
    for(l = 0; l < noutSlice; l++)
      gradBias_data[k] += scale*ptr_gradOutput[l];
  }

  /* gradient to kernels */
  THOmpLab_(conv2DRevger)(gradWeight, 1.0, scale, input, gradOutput, dH, dW);

  return 0;
}

static const struct luaL_Reg nnOmp_(SpatialConvolutionstuff__) [] = {
  {"SpatialConvolution_forwardOmp", nnOmp_(SpatialConvolution_forwardOmp)},
  {"SpatialConvolution_backwardOmp", nnOmp_(SpatialConvolution_backwardOmp)},
  {"SpatialConvolution_accGradParametersOmp", nnOmp_(SpatialConvolution_accGradParametersOmp)},
  {NULL, NULL}
};

static void nnOmp_(SpatialConvolution_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  lua_getfield(L,-1,"nn");
  luaL_register(L, NULL, nnOmp_(SpatialConvolutionstuff__));
  lua_pop(L,1);
}

#endif
