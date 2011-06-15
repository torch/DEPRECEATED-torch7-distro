#include "luaT.h"
#include "TH.h"

static const void* torch_Tensor_id = NULL;
static const void* nn_TemporalSubSampling_id = NULL;

static int nn_TemporalSubSampling_forward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int inputFrameSize = luaT_getfieldcheckint(L, 1, "inputFrameSize");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor_id);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor_id);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);

  THTensor *unfoldedInput, *unfoldedInputFrame, *unfoldedInputFrames;
  THTensor *outputFrame;
  int nInputFrame, nOutputFrame;
  int i, k;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D tensor expected");
  luaL_argcheck(L, input->size[0] == inputFrameSize, 2, "invalid input frame size");
  luaL_argcheck(L, input->size[1] >= kW, 2, "input sequence smaller than kernel size");

  nInputFrame = input->size[1];
  nOutputFrame = (nInputFrame - kW) / dW + 1;

  THTensor_resize2d(output,
                    inputFrameSize, 
                    nOutputFrame);

  outputFrame = THTensor_new();
  unfoldedInput = THTensor_new();
  unfoldedInputFrame = THTensor_new();
  unfoldedInputFrames = THTensor_new();

  THTensor_unfold(unfoldedInput, input, 1, kW, dW);
  for(k = 0; k < nOutputFrame; k++)
  {
    THTensor_select(unfoldedInputFrames, unfoldedInput, 1, k);
    THTensor_select(outputFrame, output, 1, k);
    THTensor_zero(outputFrame);
    for(i = 0; i < kW; i++)
    {
      THTensor_select(unfoldedInputFrame, unfoldedInputFrames, 1, i);
      THTensor_addTensor(outputFrame, 1, unfoldedInputFrame);
    }
    THTensor_cmul(outputFrame, weight);
    THTensor_addTensor(outputFrame, 1, bias);
  }

  THTensor_free(outputFrame);
  THTensor_free(unfoldedInput);
  THTensor_free(unfoldedInputFrame);
  THTensor_free(unfoldedInputFrames);

  return 1;
}

static int nn_TemporalSubSampling_backward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor_id);  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor_id);
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor_id);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor_id);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);

  THTensor *unfoldedInput, *unfoldedInputFrame, *unfoldedInputFrames;
  THTensor *unfoldedGradInput, *unfoldedGradInputFrame, *unfoldedGradInputFrames;
  THTensor *gradOutputFrame;
  THTensor *buffer;
  int i, k;

  buffer = THTensor_newWithSize1d(input->size[0]);
  unfoldedInput = THTensor_new();
  gradOutputFrame = THTensor_new();
  unfoldedGradInput = THTensor_new();
  unfoldedInputFrame = THTensor_new();
  unfoldedInputFrames = THTensor_new();
  unfoldedGradInputFrame = THTensor_new();
  unfoldedGradInputFrames = THTensor_new();

  /* gradWeight */
  THTensor_unfold(unfoldedInput, input, 1, kW, dW);

  /* gradInput */
  THTensor_resizeAs(gradInput, input);
  THTensor_zero(gradInput);
  THTensor_unfold(unfoldedGradInput, gradInput, 1, kW, dW);
  for(k = 0; k < gradOutput->size[1]; k++)
  {
    THTensor_select(unfoldedInputFrames, unfoldedInput, 1, k);
    THTensor_select(gradOutputFrame, gradOutput, 1, k);
    THTensor_select(unfoldedGradInputFrames, unfoldedGradInput, 1, k);
    THTensor_zero(buffer);
    for(i = 0; i < kW; i++)
    {
      THTensor_select(unfoldedInputFrame, unfoldedInputFrames, 1, i);
      THTensor_addTensor(buffer, 1, unfoldedInputFrame);

      /* gradInput */
      /* Not necessary with partial backprop: */
      THTensor_select(unfoldedGradInputFrame, unfoldedGradInputFrames, 1, i);
      THTensor_addcmul(unfoldedGradInputFrame, 1, gradOutputFrame, weight);
    }
    THTensor_cmul(buffer, gradOutputFrame);
    THTensor_addTensor(gradWeight, 1, buffer);
    THTensor_addTensor(gradBias, 1, gradOutputFrame);
  }

  THTensor_free(buffer);
  THTensor_free(unfoldedInput);
  THTensor_free(gradOutputFrame);
  THTensor_free(unfoldedGradInput);
  THTensor_free(unfoldedInputFrame);
  THTensor_free(unfoldedInputFrames);
  THTensor_free(unfoldedGradInputFrame);
  THTensor_free(unfoldedGradInputFrames);

  return 1;
}

static const struct luaL_Reg nn_TemporalSubSampling__ [] = {
  {"forward", nn_TemporalSubSampling_forward},
  {"backward", nn_TemporalSubSampling_backward},
  {NULL, NULL}
};

void nn_TemporalSubSampling_init(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  nn_TemporalSubSampling_id = luaT_newmetatable(L, "nn.TemporalSubSampling", NULL, NULL, NULL, NULL);
  luaL_register(L, NULL, nn_TemporalSubSampling__);
  lua_pop(L, 1);
}
