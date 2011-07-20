#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalConvolution.c"
#else

static int nn_(TemporalConvolution_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int inputFrameSize = luaT_getfieldcheckint(L, 1, "inputFrameSize");
  int outputFrameSize = luaT_getfieldcheckint(L, 1, "outputFrameSize");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  THTensor *outputFrame, *inputWindow;
  int nInputFrame, nOutputFrame;
  long k;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D tensor expected");
  luaL_argcheck(L, input->size[1] == inputFrameSize, 2, "invalid input frame size");
  luaL_argcheck(L, input->size[0] >= kW, 2, "input sequence smaller than kernel size");

  input = THTensor_(newContiguous)(input);
  outputFrame = THTensor_(new)();
  inputWindow = THTensor_(new)();

  nInputFrame = input->size[0];
  nOutputFrame = (nInputFrame - kW) / dW + 1;

  THTensor_(resize2d)(output,
                      nOutputFrame,
                      outputFrameSize);
  
  for(k = 0; k < nOutputFrame; k++)
  {
    THTensor_(setStorage1d)(inputWindow, input->storage, input->storageOffset+k*dW*input->size[1], kW*input->size[1], 1);
    THTensor_(select)(outputFrame, output, 0, k);
    THTensor_(copy)(outputFrame, bias);
    THTensor_(addmv)(outputFrame, 1, weight, inputWindow);
  }

  THTensor_(free)(outputFrame);
  THTensor_(free)(inputWindow);
  THTensor_(free)(input);

  return 1;
}

static int nn_(TemporalConvolution_forward2)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int inputFrameSize = luaT_getfieldcheckint(L, 1, "inputFrameSize");
  int outputFrameSize = luaT_getfieldcheckint(L, 1, "outputFrameSize");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  THTensor *outputWindow, *inputWindow;
  int nInputFrame, nOutputFrame;
  long k;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D tensor expected");
  luaL_argcheck(L, input->size[1] == inputFrameSize, 2, "invalid input frame size");
  luaL_argcheck(L, input->size[0] >= kW, 2, "input sequence smaller than kernel size");

  input = THTensor_(newContiguous)(input);
  outputWindow = THTensor_(new)();
  inputWindow = THTensor_(new)();

  nInputFrame = input->size[0];
  nOutputFrame = (nInputFrame - kW) / dW + 1;

  THTensor_(resize2d)(output,
                      nOutputFrame,
                      outputFrameSize);

  /* bias first */
  for(k = 0; k < nOutputFrame; k++)
  {
    THTensor_(select)(outputWindow, output, 0, k);
    THTensor_(copy)(outputWindow, bias);
  }
  
  
  for(k = 0; (k < kW+dW-1) && (nOutputFrame > 0); k++)
  {
    long nDistinctInputFrame = (nInputFrame+dW-1)/(kW+dW-1);
    long nFrame = THMin(nDistinctInputFrame, nOutputFrame);
    long nOverlapFrame = THMax(1, kW-dW+1);

    nOutputFrame -= nFrame;

    THTensor_(setStorage2d)(inputWindow, input->storage,
                            input->storageOffset+k*dW*input->size[1],
                            nFrame, (kW+dW-1)*input->size[1],
                            kW*input->size[1], 1);

    THTensor_(setStorage2d)(outputWindow, output->storage, 
                            output->storageOffset + k*output->size[1],
                            nFrame, nOverlapFrame*output->size[1],
                            output->size[1], 1);

//    printf("outputWindow %ld x %ld\n", outputWindow->size[0], outputWindow->size[1]);
//    printf("weight %ld x %ld\n", weight->size[0], weight->size[1]);
//    printf("inputWindow %ld x %ld\n", inputWindow->size[0], inputWindow->size[1]);

    THTensor_(transpose)(weight, NULL, 0, 1);
    THTensor_(addmm)(outputWindow, 1, inputWindow, weight);
    THTensor_(transpose)(weight, NULL, 0, 1);
  }

  THTensor_(free)(outputWindow);
  THTensor_(free)(inputWindow);
  THTensor_(free)(input);

  return 1;
}

static int nn_(TemporalConvolution_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor *gradOutputFrame;
  THTensor *inputWindow, *gradInputWindow;
  long k;


  /* Not necessary with partial backprop: */
  input = THTensor_(newContiguous)(input);
  gradOutputFrame = THTensor_(new)();
  inputWindow = THTensor_(new)();
  gradInputWindow = THTensor_(new)();

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  for(k = 0; k < gradOutput->size[0]; k++)
  {
    /* ------------------------- gradWeight ------------------------------------- */
    THTensor_(setStorage1d)(inputWindow, input->storage, input->storageOffset+k*dW*input->size[1], kW*input->size[1], 1);
    THTensor_(select)(gradOutputFrame, gradOutput, 0, k);
    THTensor_(cadd)(gradBias, 1, gradOutputFrame);
    THTensor_(addr)(gradWeight, 1, gradOutputFrame, inputWindow);

    /* -------------------------- gradInput ------------------------------------- */
    THTensor_(setStorage1d)(gradInputWindow, gradInput->storage, gradInput->storageOffset+k*dW*gradInput->size[1], kW*gradInput->size[1], 1);
    THTensor_(transpose)(weight, NULL, 0, 1);
    THTensor_(addmv)(gradInputWindow, 1, weight, gradOutputFrame);
    THTensor_(transpose)(weight, NULL, 0, 1);
  }

  THTensor_(free)(gradOutputFrame);
  THTensor_(free)(inputWindow);
  THTensor_(free)(gradInputWindow);
  THTensor_(free)(input);

  return 1;
}

static const struct luaL_Reg nn_(TemporalConvolution__) [] = {
  {"TemporalConvolution_forward", nn_(TemporalConvolution_forward)},
  {"TemporalConvolution_forward2", nn_(TemporalConvolution_forward2)},
  {"TemporalConvolution_backward", nn_(TemporalConvolution_backward)},
  {NULL, NULL}
};

static void nn_(TemporalConvolution_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(TemporalConvolution__), "nn");
  lua_pop(L,1);
}

#endif
