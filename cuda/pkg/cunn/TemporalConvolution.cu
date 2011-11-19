static int cunn_TemporalConvolution_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, torch_CudaTensor_id);  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int inputFrameSize = luaT_getfieldcheckint(L, 1, "inputFrameSize");
  int outputFrameSize = luaT_getfieldcheckint(L, 1, "outputFrameSize");

  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", torch_CudaTensor_id);
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", torch_CudaTensor_id);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", torch_CudaTensor_id);

  THCudaTensor *outputWindow, *inputWindow;
  int nInputFrame, nOutputFrame;
  long k;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D tensor expected");
  luaL_argcheck(L, input->size[1] == inputFrameSize, 2, "invalid input frame size");
  luaL_argcheck(L, input->size[0] >= kW, 2, "input sequence smaller than kernel size");

  input = THCudaTensor_newContiguous(input);
  outputWindow = THCudaTensor_new();
  inputWindow = THCudaTensor_new();

  nInputFrame = input->size[0];
  nOutputFrame = (nInputFrame - kW) / dW + 1;

  THCudaTensor_resize2d(output,
                      nOutputFrame,
                      outputFrameSize);

  /* bias first */
  for(k = 0; k < nOutputFrame; k++)
  {
    THCudaTensor_select(outputWindow, output, 0, k);
    THCudaTensor_copy(outputWindow, bias);
  }
  

  /* ouch */
  for(k = 0; nOutputFrame > 0; k++)
  {
    long outputFrameStride = (kW-1)/dW+1;
    long inputFrameStride = outputFrameStride*dW;
    long nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
    nOutputFrame -= nFrame;

    THCudaTensor_setStorage2d(inputWindow, input->storage,
                            input->storageOffset+k*dW*input->size[1],
                            nFrame, inputFrameStride*input->size[1],
                            kW*input->size[1], 1);

    THCudaTensor_setStorage2d(outputWindow, output->storage, 
                            output->storageOffset + k*output->size[1],
                            nFrame, outputFrameStride*output->size[1],
                            output->size[1], 1);

    THCudaTensor_transpose(weight, NULL, 0, 1);
    THCudaTensor_addmm(outputWindow, 1, 1, inputWindow, weight);
    THCudaTensor_transpose(weight, NULL, 0, 1);
  }

  THCudaTensor_free(outputWindow);
  THCudaTensor_free(inputWindow);
  THCudaTensor_free(input);

  return 1;
}

static int cunn_TemporalConvolution_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, torch_CudaTensor_id);  
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, torch_CudaTensor_id);  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  long nInputFrame = input->size[0];
  long nOutputFrame = gradOutput->size[0];

  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", torch_CudaTensor_id);
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", torch_CudaTensor_id);

  THCudaTensor *gradOutputWindow;
  THCudaTensor *gradInputWindow;
  long k;


  /* Not necessary with partial backprop: */
  gradOutputWindow = THCudaTensor_new();
  gradInputWindow = THCudaTensor_new();

  THCudaTensor_resizeAs(gradInput, input);
  THCudaTensor_zero(gradInput);

  /* ouch */
  for(k = 0; nOutputFrame > 0; k++)
  {
    long outputFrameStride = (kW-1)/dW+1;
    long inputFrameStride = outputFrameStride*dW;
    long nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
    nOutputFrame -= nFrame;

    THCudaTensor_setStorage2d(gradOutputWindow, gradOutput->storage, 
                            gradOutput->storageOffset + k*gradOutput->size[1],
                            nFrame, outputFrameStride*gradOutput->size[1],
                            gradOutput->size[1], 1);

    THCudaTensor_setStorage2d(gradInputWindow, gradInput->storage,
                            gradInput->storageOffset+k*dW*gradInput->size[1],
                            nFrame, inputFrameStride*gradInput->size[1],
                            kW*gradInput->size[1], 1);

    THCudaTensor_addmm(gradInputWindow, 1, 1, gradOutputWindow, weight);
  }

  THCudaTensor_free(gradOutputWindow);
  THCudaTensor_free(gradInputWindow);

  return 1;
}

static int cunn_TemporalConvolution_accGradParameters(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, torch_CudaTensor_id);  
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, torch_CudaTensor_id);  
  float scale = luaL_optnumber(L, 4, 1);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  long nInputFrame = input->size[0];
  long nOutputFrame = gradOutput->size[0];

  THCudaTensor *gradWeight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradWeight", torch_CudaTensor_id);
  THCudaTensor *gradBias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradBias", torch_CudaTensor_id);

  THCudaTensor *gradOutputWindow;
  THCudaTensor *inputWindow;
  long k;


  /* Not necessary with partial backprop: */
  input = THCudaTensor_newContiguous(input);
  gradOutputWindow = THCudaTensor_new();
  inputWindow = THCudaTensor_new();

  /* bias first */
  for(k = 0; k < nOutputFrame; k++)
  {
    THCudaTensor_select(gradOutputWindow, gradOutput, 0, k);
    THCudaTensor_cadd(gradBias, scale, gradOutputWindow);
  }

  /* ouch */
  for(k = 0; nOutputFrame > 0; k++)
  {
    long outputFrameStride = (kW-1)/dW+1;
    long inputFrameStride = outputFrameStride*dW;
    long nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
    nOutputFrame -= nFrame;

    THCudaTensor_setStorage2d(inputWindow, input->storage,
                            input->storageOffset+k*dW*input->size[1],
                            nFrame, inputFrameStride*input->size[1],
                            kW*input->size[1], 1);

    THCudaTensor_setStorage2d(gradOutputWindow, gradOutput->storage, 
                            gradOutput->storageOffset + k*gradOutput->size[1],
                            nFrame, outputFrameStride*gradOutput->size[1],
                            gradOutput->size[1], 1);

    THCudaTensor_transpose(gradOutputWindow, NULL, 0, 1);
    THCudaTensor_addmm(gradWeight, 1, scale, gradOutputWindow, inputWindow);
    THCudaTensor_transpose(gradOutputWindow, NULL, 0, 1);
  }

  THCudaTensor_free(gradOutputWindow);
  THCudaTensor_free(inputWindow);
  THCudaTensor_free(input);

  return 1;
}

static const struct luaL_Reg cunn_TemporalConvolution__ [] = {
  {"TemporalConvolution_updateOutput", cunn_TemporalConvolution_updateOutput},
  {"TemporalConvolution_updateGradInput", cunn_TemporalConvolution_updateGradInput},
  {"TemporalConvolution_accGradParameters", cunn_TemporalConvolution_accGradParameters},
  {NULL, NULL}
};

static void cunn_TemporalConvolution_init(lua_State *L)
{
  luaT_pushmetaclass(L, torch_CudaTensor_id);
  luaT_registeratname(L, cunn_TemporalConvolution__, "nn");
  lua_pop(L,1);
}
