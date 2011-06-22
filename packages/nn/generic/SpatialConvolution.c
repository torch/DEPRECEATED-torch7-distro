#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolution.c"
#else

static int nn_(SpatialConvolution_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  real *weight_data = THTensor_(data)(weight);
  real *bias_data = THTensor_(data)(bias);
  real *output_data;
  real *input_data;

  long i, k;
  long inputWidth, inputHeight, outputWidth, outputHeight;

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");

  inputWidth = input->size[2];
  inputHeight = input->size[1];
  outputWidth = (inputWidth - kW) / dW + 1;
  outputHeight = (inputHeight - kH) / dH + 1;

  luaL_argcheck(L, input->size[0] == nInputPlane, 2, "invalid number of input planes");
  luaL_argcheck(L, inputWidth >= kW && inputHeight >= kH, 2, "input image smaller than kernel size");

  input = THTensor_(newContiguous)(input, 0);
  input_data = THTensor_(data)(input);

  THTensor_(resize3d)(output,
                      nOutputPlane,
                      outputHeight,
                      outputWidth);

  output_data = THTensor_(data)(output);

  for(k = 0; k < nOutputPlane; k++)
  {
    real z = bias_data[k];
    for(i = 0; i < outputWidth*outputHeight; i++)
      output_data[i] = z;

    for(i = 0; i < nInputPlane; i++)
    {
      long xx, yy;

      /* Get the good mask for (k,i) (k out, i in) */
      real *ptr_weight = weight_data+k*weight->stride[0]+i*kW*kH;
      
      /* Get the input image */
      real *ptr_input = input_data+i*inputWidth*inputHeight;
      
      /* For all output pixels... */
      real *ptr_output = output_data;
      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          /* Dot product in two dimensions... (between input image and the mask) */
          real *ptr_input_ = ptr_input+yy*dH*inputWidth+xx*dW;
          real *ptr_weight_ = ptr_weight;
          real sum = 0;
          long kx, ky;
          for(ky = 0; ky < kH; ky++)
          {
            for(kx = 0; kx < kW; kx++)
              sum += ptr_input_[kx]*ptr_weight_[kx];
            ptr_input_ += inputWidth; /* next input line */
            ptr_weight_ += kW; /* next mask line */
          }
          
          /* Update output */
          *ptr_output++ += sum;
        }
      }
    }

    /* Next output plane */
    output_data += outputWidth*outputHeight;
  }

  THTensor_(free)(input);

  return 1;
}

static int nn_(SpatialConvolution_forward2)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");

  long nOutputPlane = weight->size[0];
  long nInputPlane  = weight->size[1];
  long kW           = weight->size[2];
  long kH           = weight->size[3];
  long inputWidth   = input->size[2];
  long inputHeight  = input->size[1];
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;

  THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);

  /* add bias */
  THArgCheck(bias->size[0] == nOutputPlane, 1, "Number of output features is not same as number of bias elements");
  long i;
  THTensor *outn = THTensor_(new)();
  real* bias_data = THTensor_(data)(bias);
  for (i=0; i<bias->size[0]; i++) {
    THTensor_(select)(outn,output,0,i);
    THTensor_(add)(outn,bias_data[i]);
  }
  THTensor_(free)(outn);

  /* do convolutions */
  THLab_(conv2Dmv)(output, 1.0, input, weight, dH, dW, "valid");

  return 1;
}


static int nn_(SpatialConvolution_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));
  
  long inputWidth = input->size[2];
  long inputHeight = input->size[1];
  long outputWidth = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;

  long i, k;

  real *weight_data = THTensor_(data)(weight);
  real *gradWeight_data = THTensor_(data)(gradWeight);
  real *gradBias_data = THTensor_(data)(gradBias);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  real *input_data, *gradInput_data;

  input = THTensor_(newContiguous)(input, 0);
  input_data = THTensor_(data)(input);

  for(k = 0; k < nOutputPlane; k++)
  {
    real sum = 0;
    for(i = 0; i < outputWidth*outputHeight; i++)
      sum += gradOutput_data[i];
    gradBias_data[k] += sum;

    for(i = 0; i < nInputPlane; i++)
    {
      real *ptr_gradWeight = gradWeight_data+k*gradWeight->stride[0] + i*kW*kH;
      real *ptr_input = input_data+i*inputWidth*inputHeight;
      real *ptr_gradOutput = gradOutput_data;
      long xx, yy;

      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          real *ptr_input_ = ptr_input+yy*dH*inputWidth+xx*dW;
          real *ptr_gradWeight_ = ptr_gradWeight;          
          real z = *ptr_gradOutput++;
          long kx, ky;

          for(ky = 0; ky < kH; ky++)
          {
            for(kx = 0; kx < kW; kx++)
              ptr_gradWeight_[kx] += z * ptr_input_[kx];
            ptr_input_ += inputWidth;
            ptr_gradWeight_ += kW;
          }
        }
      }
    }
    gradOutput_data += outputWidth*outputHeight;
  }

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);  
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);

  for(k = 0; k < nOutputPlane; k++)
  {
    for(i = 0; i < nInputPlane; i++)
    {
      real *ptr_weight = weight_data+k*weight->stride[0]+i*kW*kH;
      real *ptr_gradInput = gradInput_data+i*inputWidth*inputHeight;
      real *ptr_gradOutput = gradOutput_data;
      long xx, yy;

      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          real *ptr_gradInput_ = ptr_gradInput+yy*dH*inputWidth+xx*dW;
          real *ptr_weight_ = ptr_weight;          
          real z = *ptr_gradOutput++;
          long kx, ky;

          for(ky = 0; ky < kH; ky++)
          {
            for(kx = 0; kx < kW; kx++)
              ptr_gradInput_[kx] += z * ptr_weight_[kx];
            ptr_gradInput_ += inputWidth;
            ptr_weight_ += kW;
          }
        }
      }
    }
    gradOutput_data += outputWidth*outputHeight;
  }

  THTensor_(free)(input);

  return 1;
}

static int nn_(SpatialConvolution_backward2)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));
  
  long inputWidth = input->size[2];
  long inputHeight = input->size[1];
  long outputWidth = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;

  long k;

  /* gradient to bias */
  real *gradBias_data = THTensor_(data)(gradBias);
  THTensor* gradOutSlice = THTensor_(new)();
  for(k = 0; k < nOutputPlane; k++)
  {
    THTensor_(select)(gradOutSlice, gradOutput, 0, k);
    gradBias_data[k] += THTensor_(sum)(gradOutSlice);
  }
  THTensor_(free)(gradOutSlice);

  /* gradient to kernels */
  THLab_(conv2Dger)(gradWeight, 1.0, input, gradOutput, dH, dW, "valid");

  /* gradient to input */
  THTensor *tweight = THTensor_(newTranspose)(weight,0,1);
  THLab_(conv2Dmv)(gradInput, 0.0, gradOutput, tweight, dH, dW, "full");
  THTensor_(free)(tweight);

  return 1;
}

static const struct luaL_Reg nn_(SpatialConvolution__) [] = {
  {"SpatialConvolution_forward", nn_(SpatialConvolution_forward)},
  {"SpatialConvolution_backward", nn_(SpatialConvolution_backward)},
  {"SpatialConvolution_forward2", nn_(SpatialConvolution_forward2)},
  {"SpatialConvolution_backward2", nn_(SpatialConvolution_backward2)},
  {NULL, NULL}
};

static void nn_(SpatialConvolution_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SpatialConvolution__), "nn");
  lua_pop(L,1);
}

#endif
