#include "luaT.h"
#include "TH.h"

static const void* torch_Tensor_id = NULL;
static const void* nn_SpatialSubSampling_id = NULL;

static int nn_SpatialSubSampling_forward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor_id);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor_id);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);

  THTensor *outputPlane, *inputPlane, *weightPlane, *unfoldedInputPlane;
  int k;
  
  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");
  luaL_argcheck(L, input->size[2] == nInputPlane, 2, "invalid number of input planes");
  luaL_argcheck(L, input->size[0] >= kW && input->size[1] >= kH, 2, "input image smaller than kernel size");

  THTensor_resize3d(output,
                    (input->size[0] - kW) / dW + 1, 
                    (input->size[1] - kH) / dH + 1,
                    nInputPlane);

  inputPlane = THTensor_new();
  weightPlane = THTensor_new();
  outputPlane = THTensor_new();
  unfoldedInputPlane = THTensor_new();
  
  for(k = 0; k < nInputPlane; k++)
  {
    THTensor_select(inputPlane,  input,  2, k);
    THTensor_select(outputPlane, output, 2, k);
    
    /* Initialize to the bias */
    THTensor_fill(outputPlane, THTensor_get1d(bias, k));

    /* Get the input image */
    THTensor_unfold(unfoldedInputPlane, inputPlane, 0, kW, dW);
    THTensor_unfold(unfoldedInputPlane, NULL,       1, kH, dH);

    THTensor_setStorage2d(weightPlane, weight->storage, weight->storageOffset+weight->stride[0]*k, kW, 0, kH, 0);
    THTensor_addT4dotT2(outputPlane, 1, unfoldedInputPlane, weightPlane);
  }

  THTensor_free(inputPlane);
  THTensor_free(weightPlane);
  THTensor_free(outputPlane);
  THTensor_free(unfoldedInputPlane);

  return 1;
}

static int nn_SpatialSubSampling_backward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor_id);  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor_id);
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor_id);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor_id);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);

  THTensor *gradInputPlane, *unfoldedInputPlane, *unfoldedGradInputPlane, *inputPlane;
  THTensor *gradOutputPlane;
  THTensor *weightPlane, *gradWeightPlane;
  int k;

  gradInputPlane = THTensor_new();
  unfoldedInputPlane = THTensor_new();
  unfoldedGradInputPlane = THTensor_new();
  inputPlane = THTensor_new();
  gradOutputPlane = THTensor_new();
  weightPlane = THTensor_new();
  gradWeightPlane = THTensor_new();
  
  /* Not necessary with partial backprop: */
  THTensor_resizeAs(gradInput, input);
  THTensor_zero(gradInput);

  for(k = 0; k < nInputPlane; k++)
  {
    THTensor_select(gradOutputPlane, gradOutput, 2, k);
    THTensor_set1d(gradBias, k, THTensor_get1d(gradBias, k) + THTensor_sum(gradOutputPlane));
      
    /* ------------------------- gradWeight ------------------------------------- */
    
    /* Get the input image */
    THTensor_select(inputPlane, input, 2, k);
    THTensor_unfold(unfoldedInputPlane, inputPlane, 0, kW, dW);
    THTensor_unfold(unfoldedInputPlane, NULL,       1, kH, dH);
    THTensor_transpose(unfoldedInputPlane,NULL,0,2);
    THTensor_transpose(unfoldedInputPlane,NULL,1,3);

    THTensor_setStorage2d(gradWeightPlane, gradWeight->storage, gradWeight->storageOffset+gradWeight->stride[0]*k, kW, 0, kH, 0);
    THTensor_addT4dotT2(gradWeightPlane, 1, unfoldedInputPlane, gradOutputPlane);

    /* -------------------------- gradInput ------------------------------------- */
    
    /* Not necessary with partial backprop: */
    
    /* Get the gradInput image */
    THTensor_select(gradInputPlane, gradInput, 2, k);
    THTensor_unfold(unfoldedGradInputPlane, gradInputPlane, 0, kW, dW);
    THTensor_unfold(unfoldedGradInputPlane, NULL,           1, kH, dH);
    
    THTensor_setStorage2d(weightPlane, weight->storage, weight->storageOffset+weight->stride[0]*k, kW, 0, kH, 0);
    THTensor_addT2outT2(unfoldedGradInputPlane, 1, gradOutputPlane, weightPlane);
  }

  THTensor_free(gradInputPlane);
  THTensor_free(unfoldedInputPlane);
  THTensor_free(unfoldedGradInputPlane);
  THTensor_free(inputPlane);
  THTensor_free(gradOutputPlane);
  THTensor_free(weightPlane);
  THTensor_free(gradWeightPlane);

  return 1;
}

static const struct luaL_Reg nn_SpatialSubSampling__ [] = {
  {"forward", nn_SpatialSubSampling_forward},
  {"backward", nn_SpatialSubSampling_backward},
  {NULL, NULL}
};

void nn_SpatialSubSampling_init(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  nn_SpatialSubSampling_id = luaT_newmetatable(L, "nn.SpatialSubSampling", NULL, NULL, NULL, NULL);
  luaL_register(L, NULL, nn_SpatialSubSampling__);
  lua_pop(L, 1);
}
