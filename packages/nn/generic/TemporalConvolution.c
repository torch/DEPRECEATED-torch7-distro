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

  /* ouch */
  for(k = 0; (k < kW+dW-1) && (nOutputFrame > 0); k++)
  {
    long nOverlapFrame = (kW-1)/dW+1;
    long nGapFrame = nOverlapFrame*dW;
    long nFrame = (nInputFrame-k*dW-kW)/nGapFrame + 1;
//    long nOverlapFrame = ((kW % dW) ? kW / dW + 1 : kW / dW);
//    long nGapFrame = ((kW % dW) ? kW / dW + 1 : kW / dW)*dW;
    nOutputFrame -= nFrame;

//    printf("k = %ld, gap = %ld nframe = %ld\n", k, nGapFrame, nFrame);

//    printf("k = %ld, input->storage->size: %ld\n", k, input->storage->size);
    THTensor_(setStorage2d)(inputWindow, input->storage,
                            input->storageOffset+k*dW*input->size[1],
                            nFrame, nGapFrame*input->size[1],
                            kW*input->size[1], 1);
//    printf("k = %ld, input->storage->size: %ld\n", k, input->storage->size);

//    printf("k = %ld, output->storage->size: %ld\n", k, output->storage->size);
    THTensor_(setStorage2d)(outputWindow, output->storage, 
                            output->storageOffset + k*output->size[1],
                            nFrame, nOverlapFrame*output->size[1],
                            output->size[1], 1);
//    printf("k = %ld, output->storage->size: %ld\n", k, output->storage->size);

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

static int nn_(TemporalConvolution_backward2)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  long nInputFrame = input->size[0];
  long nOutputFrame = gradOutput->size[0];

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor *gradOutputWindow;
  THTensor *inputWindow, *gradInputWindow;
  long k;


  /* Not necessary with partial backprop: */
  input = THTensor_(newContiguous)(input);
  gradOutputWindow = THTensor_(new)();
  inputWindow = THTensor_(new)();
  gradInputWindow = THTensor_(new)();

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  /* bias first */
  for(k = 0; k < nOutputFrame; k++)
  {
    THTensor_(select)(gradOutputWindow, gradOutput, 0, k);
    THTensor_(cadd)(gradBias, 1, gradOutputWindow);
  }

  /* ouch */
  for(k = 0; (k < kW+dW-1) && (nOutputFrame > 0); k++)
  {
    long nOverlapFrame = (kW-1)/dW+1;
    long nGapFrame = nOverlapFrame*dW;
    long nFrame = (nInputFrame-k*dW-kW)/nGapFrame + 1;

//X    long nGapFrame = ((kW % dW) ? kW / dW + 1 : kW / dW)*dW;
//X    long nFrame = (nInputFrame-k-kW)/nGapFrame + 1;
//    long nDistinctInputFrame = (nInputFrame-k+THMax(0,dW-kW))/(kW+THMax(0,dW-kW));//(nInputFrame-k+dW-1)/(kW+dW-1);
//    long nDistinctInputFrame = (nInputFrame-k+dW-1)/(kW+dW-1);
//    long nFrame = THMin(nDistinctInputFrame, nOutputFrame);
//X    long nOverlapFrame = THMax(1, kW-dW+1);
    nOutputFrame -= nFrame;

    /* ------------------------- gradWeight ------------------------------------- */

//    printf("k = %ld, input->storage->size: %ld\n", k, input->storage->size);
//    printf("nframe = %ld\n", nFrame);
//    printf("ok1\n");
    THTensor_(setStorage2d)(inputWindow, input->storage,
                            input->storageOffset+k*dW*input->size[1],
                            nFrame, nGapFrame*input->size[1],
                            kW*input->size[1], 1);
//    printf("k = %ld, input->storage->size: %ld\n", k, input->storage->size);
//    printf("k = %ld, gradOutput->storage->size: %ld\n", k, gradOutput->storage->size);

    
//    printf("ok2\n");
    THTensor_(setStorage2d)(gradOutputWindow, gradOutput->storage, 
                            gradOutput->storageOffset + k*gradOutput->size[1],
                            nFrame, nOverlapFrame*gradOutput->size[1],
                            gradOutput->size[1], 1);

//    printf("ok3 k=%ld nframe=%ld\n", k, nFrame);
    THTensor_(transpose)(gradOutputWindow, NULL, 0, 1);
//    printf("ok4\n");
    THTensor_(addmm)(gradWeight, 1, gradOutputWindow, inputWindow);
//    printf("ok5\n");
    THTensor_(transpose)(gradOutputWindow, NULL, 0, 1);

//    printf("k = %ld, gradOutput->storage->size: %ld\n", k, gradOutput->storage->size);

    /* -------------------------- gradInput ------------------------------------- */

//    printf("ok6\n");
    THTensor_(setStorage2d)(gradInputWindow, gradInput->storage,
                            gradInput->storageOffset+k*dW*gradInput->size[1],
                            nFrame, nGapFrame*gradInput->size[1],
                            kW*gradInput->size[1], 1);

//    printf("ok7\n");
    THTensor_(addmm)(gradInputWindow, 1, gradOutputWindow, weight);

//    printf("ok8\n");

  }

  THTensor_(free)(gradOutputWindow);
  THTensor_(free)(inputWindow);
  THTensor_(free)(gradInputWindow);
  THTensor_(free)(input);

  return 1;
}

static const struct luaL_Reg nn_(TemporalConvolution__) [] = {
  {"TemporalConvolution_forward", nn_(TemporalConvolution_forward)},
  {"TemporalConvolution_forward2", nn_(TemporalConvolution_forward2)},
  {"TemporalConvolution_backward", nn_(TemporalConvolution_backward)},
  {"TemporalConvolution_backward2", nn_(TemporalConvolution_backward2)},
  {NULL, NULL}
};

static void nn_(TemporalConvolution_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(TemporalConvolution__), "nn");
  lua_pop(L,1);
}

#endif
