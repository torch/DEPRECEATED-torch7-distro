#include "luaT.h"
#include "TH.h"

static const void* torch_Tensor_id = NULL;
static const void* nn_MSECriterion_id = NULL;

static int nn_MSECriterion_forward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);  
  THTensor *target = luaT_checkudata(L, 3, torch_Tensor_id);  
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  double sum;

  sum = 0;
  TH_TENSOR_APPLY2(double, input, double, target,
                   double z = (*input_p - *target_p);
                   sum += z*z;)

  if(sizeAverage)
    sum /= THTensor_nElement(input);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}

static int nn_MSECriterion_backward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor *target = luaT_checkudata(L, 3, torch_Tensor_id);
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);
  double norm = (sizeAverage ? 2./((double)THTensor_nElement(input)) : 2.);

  THTensor_resizeAs(gradInput, input);
  TH_TENSOR_APPLY3(double, gradInput, double, input, double, target,
                   *gradInput_p = norm * (*input_p - *target_p);)
  return 1;
}

static const struct luaL_Reg nn_MSECriterion__ [] = {
  {"forward", nn_MSECriterion_forward},
  {"backward", nn_MSECriterion_backward},
  {NULL, NULL}
};

void nn_MSECriterion_init(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  nn_MSECriterion_id = luaT_newmetatable(L, "nn.MSECriterion", NULL, NULL, NULL, NULL);
  luaL_register(L, NULL, nn_MSECriterion__);
  lua_pop(L, 1);
}
