#include "luaT.h"
#include "TH.h"

static const void* torch_Tensor_id = NULL;
static const void* nn_HardTanh_id = NULL;

static int nn_HardTanh_forward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);

  THTensor_resizeAs(output, input);

  TH_TENSOR_APPLY2(double, output, double, input, \
                   if(*input_p < -1) \
                     *output_p = -1; \
                   else if(*input_p <= 1) \
                     *output_p = *input_p; \
                   else \
                     *output_p = 1;)
  return 1;
}

static int nn_HardTanh_backward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor_id);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);

  THTensor_resizeAs(gradInput, input);
  TH_TENSOR_APPLY3(double, gradInput, double, gradOutput, double, input, \
                   if(*input_p < -1 || *input_p > 1) \
                     *gradInput_p = 0; \
                   else \
                     *gradInput_p = *gradOutput_p;);
  return 1;
}

static const struct luaL_Reg nn_HardTanh__ [] = {
  {"forward", nn_HardTanh_forward},
  {"backward", nn_HardTanh_backward},
  {NULL, NULL}
};

void nn_HardTanh_init(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  nn_HardTanh_id = luaT_newmetatable(L, "nn.HardTanh", NULL, NULL, NULL, NULL);
  luaL_register(L, NULL, nn_HardTanh__);
  lua_pop(L, 1);
}
