#include "luaT.h"
#include "TH.h"

static const void* torch_Tensor_id = NULL;
static const void* nn_TemporalConvolution_id = NULL;

static int nn_TemporalConvolution_forward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int inputFrameSize = luaT_getfieldcheckint(L, 1, "inputFrameSize");
  int outputFrameSize = luaT_getfieldcheckint(L, 1, "outputFrameSize");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor_id);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor_id);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);

  THTensor *outputFrame, *unfoldedInput, *unfoldedInputFrame, *xWeight;
  int nInputFrame, nOutputFrame;
  int k;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D tensor expected");
  luaL_argcheck(L, input->size[0] == inputFrameSize, 2, "invalid input frame size");
  luaL_argcheck(L, input->size[1] >= kW, 2, "input sequence smaller than kernel size");

  nInputFrame = input->size[1];
  nOutputFrame = (nInputFrame - kW) / dW + 1;

  THTensor_resize2d(output,
                    outputFrameSize, 
                    nOutputFrame);

  xWeight = THTensor_new();
  outputFrame = THTensor_new();
  unfoldedInput = THTensor_new();
  unfoldedInputFrame = THTensor_new();

  THTensor_unfold(unfoldedInput, input, 1, kW, dW);

  THTensor_setStorage4d(xWeight, weight->storage, weight->storageOffset, weight->size[0], weight->stride[0], weight->size[1], weight->stride[1], weight->size[2], weight->stride[2], 1, -1);
  THTensor_transpose(xWeight,NULL,0,2);
  THTensor_transpose(xWeight,NULL,1,3);

  for(k = 0; k < nOutputFrame; k++)
  {
    THTensor_select(unfoldedInputFrame, unfoldedInput, 1, k);
    THTensor_narrow(outputFrame, output, 1, k, 1);
    THTensor_copy(outputFrame, bias);
    THTensor_addT4dotT2(outputFrame, 1, xWeight, unfoldedInputFrame);
  }

  THTensor_free(xWeight);
  THTensor_free(outputFrame);
  THTensor_free(unfoldedInput);
  THTensor_free(unfoldedInputFrame);

  return 1;
}

static int nn_TemporalConvolution_backward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor_id);  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor_id);
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor_id);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor_id);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);

  THTensor *unfoldedGradInput, *unfoldedInputFrame, *unfoldedGradInputFrame, *unfoldedInput;
  THTensor *gradOutputFrame;
  THTensor *xWeight, *xGradWeight;
  int k;

  xWeight = THTensor_new();
  xGradWeight = THTensor_new();
  unfoldedGradInput = THTensor_new();
  unfoldedInputFrame = THTensor_new();
  unfoldedGradInputFrame = THTensor_new();
  unfoldedInput = THTensor_new();
  gradOutputFrame = THTensor_new();

  /* Not necessary with partial backprop: */
  THTensor_resizeAs(gradInput, input);
  THTensor_zero(gradInput);

  THTensor_setStorage4d(xWeight, weight->storage, weight->storageOffset, weight->size[0], weight->stride[0], weight->size[1], weight->stride[1], weight->size[2], weight->stride[2], 1, -1);
  THTensor_setStorage4d(xGradWeight, gradWeight->storage, gradWeight->storageOffset, gradWeight->size[0], gradWeight->stride[0], gradWeight->size[1], gradWeight->stride[1], gradWeight->size[2], gradWeight->stride[2], 1, -1);

  THTensor_unfold(unfoldedInput, input, 1, kW, dW);
  THTensor_unfold(unfoldedGradInput, gradInput, 1, kW, dW);

  for(k = 0; k < gradOutput->size[1]; k++)
  {
    /* ------------------------- gradWeight ------------------------------------- */
    THTensor_select(unfoldedInputFrame, unfoldedInput, 1, k);
    THTensor_narrow(gradOutputFrame, gradOutput, 1, k, 1);
    THTensor_addTensor(gradBias, 1, gradOutputFrame);
    THTensor_addT2outT2(xGradWeight, 1, unfoldedInputFrame, gradOutputFrame);

    /* -------------------------- gradInput ------------------------------------- */
    THTensor_select(unfoldedGradInputFrame, unfoldedGradInput, 1, k);
    THTensor_addT4dotT2(unfoldedGradInputFrame, 1, xWeight, gradOutputFrame);
  }

  THTensor_free(xWeight);
  THTensor_free(xGradWeight);
  THTensor_free(unfoldedGradInput);
  THTensor_free(unfoldedInputFrame);
  THTensor_free(unfoldedGradInputFrame);
  THTensor_free(unfoldedInput);
  THTensor_free(gradOutputFrame);

  return 1;
}

static const struct luaL_Reg nn_TemporalConvolution__ [] = {
  {"forward", nn_TemporalConvolution_forward},
  {"backward", nn_TemporalConvolution_backward},
  {NULL, NULL}
};

void nn_TemporalConvolution_init(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  nn_TemporalConvolution_id = luaT_newmetatable(L, "nn.TemporalConvolution", NULL, NULL, NULL, NULL);
  luaL_register(L, NULL, nn_TemporalConvolution__);
  lua_pop(L, 1);
}
