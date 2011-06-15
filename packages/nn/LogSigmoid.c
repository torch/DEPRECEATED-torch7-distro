#include "luaT.h"
#include "TH.h"

static const void* torch_Tensor_id = NULL;
static const void* nn_LogSigmoid_id = NULL;

static int nn_LogSigmoid_forward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor *buffer = luaT_getfieldcheckudata(L, 1, "buffer", torch_Tensor_id);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);

  THTensor_resizeAs(output, input);
  THTensor_resizeAs(buffer, input);

  TH_TENSOR_APPLY3(double, output, double, input, double, buffer, \
                   double z = exp(-*input_p); \
                   *buffer_p = z;
                   *output_p = -log(1. + z);)

  return 1;
}

static int nn_LogSigmoid_backward(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor_id);
  THTensor *buffer = luaT_getfieldcheckudata(L, 1, "buffer", torch_Tensor_id);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);

  THTensor_resizeAs(gradInput, buffer);
  TH_TENSOR_APPLY3(double, gradInput, double, gradOutput, double, buffer, \
                   double z = *buffer_p; \
                   *gradInput_p = *gradOutput_p * z / (1. + z);)

  return 1;
}

static const struct luaL_Reg nn_LogSigmoid__ [] = {
  {"forward", nn_LogSigmoid_forward},
  {"backward", nn_LogSigmoid_backward},
  {NULL, NULL}
};

void nn_LogSigmoid_init(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  nn_LogSigmoid_id = luaT_newmetatable(L, "nn.LogSigmoid", NULL, NULL, NULL, NULL);
  luaL_register(L, NULL, nn_LogSigmoid__);
  lua_pop(L, 1);
}
