#include "luaT.h"
#include "TH.h"

static const void* torch_Tensor_id = NULL;
static const void* nn_SoftPlus_id = NULL;

static int nn_SoftPlus_forward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);
  
  THTensor_resizeAs(output, input);

  TH_TENSOR_APPLY2(double, output, double, input, \
                   *output_p = log1p(exp(*input_p));)

  return 1;
}

static int nn_SoftPlus_backward(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor_id);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);

  THTensor_resizeAs(gradInput, output);
  TH_TENSOR_APPLY3(double, gradInput, double, gradOutput, double, output, \
                   double z = exp(*output_p); \
                   *gradInput_p = *gradOutput_p * (z - 1.)/z;)
  return 1;
}

static const struct luaL_Reg nn_SoftPlus__ [] = {
  {"forward", nn_SoftPlus_forward},
  {"backward", nn_SoftPlus_backward},
  {NULL, NULL}
};

void nn_SoftPlus_init(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  nn_SoftPlus_id = luaT_newmetatable(L, "nn.SoftPlus", NULL, NULL, NULL, NULL);
  luaL_register(L, NULL, nn_SoftPlus__);
  lua_pop(L, 1);
}
