#include "luaT.h"
#include "TH.h"

static const void* torch_Tensor_id = NULL;
static const void* nn_SoftMax_id = NULL;

static int nn_SoftMax_forward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);  
  double shift = luaT_getfieldchecknumber(L, 1, "shift");
  int computeShift = luaT_getfieldcheckboolean(L, 1, "computeShift");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);
  double sum;

  if(computeShift)
  {
    shift = THTensor_max(input);
    lua_pushnumber(L, shift);
    lua_setfield(L, 1, "shift");
  }

  THTensor_resizeAs(output, input);

  sum = 0;
  TH_TENSOR_APPLY2(double, output, double, input, \
                   double z = exp(*input_p - shift); \
                   *output_p = z; \
                   sum += z;)

  THTensor_mul(output, 1/sum);

  return 1;
}

static int nn_SoftMax_backward(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor_id);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);
  double sum;

  sum = THTensor_dot(gradOutput, output);
  THTensor_resizeAs(gradInput, output);
  TH_TENSOR_APPLY3(double, gradInput, double, gradOutput, double, output, \
                   *gradInput_p = *output_p * (*gradOutput_p - sum);)
  return 1;
}

static const struct luaL_Reg nn_SoftMax__ [] = {
  {"forward", nn_SoftMax_forward},
  {"backward", nn_SoftMax_backward},
  {NULL, NULL}
};

void nn_SoftMax_init(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  nn_SoftMax_id = luaT_newmetatable(L, "nn.SoftMax", NULL, NULL, NULL, NULL);
  luaL_register(L, NULL, nn_SoftMax__);
  lua_pop(L, 1);
}
