#include "luaT.h"

extern void nn_Mean_init(lua_State *L);
extern void nn_Min_init(lua_State *L); 
extern void nn_Max_init(lua_State *L);
extern void nn_Sum_init(lua_State *L);

extern void nn_Exp_init(lua_State *L);
extern void nn_HardTanh_init(lua_State *L);
extern void nn_LogSigmoid_init(lua_State *L);
extern void nn_LogSoftMax_init(lua_State *L);
extern void nn_Sigmoid_init(lua_State *L);
extern void nn_SoftMax_init(lua_State *L);
extern void nn_SoftPlus_init(lua_State *L);
extern void nn_Tanh_init(lua_State *L);

extern void nn_SpatialConvolution_init(lua_State *L);
extern void nn_SpatialSubSampling_init(lua_State *L);
extern void nn_TemporalConvolution_init(lua_State *L);
extern void nn_TemporalSubSampling_init(lua_State *L);

extern void nn_MSECriterion_init(lua_State *L);
extern void nn_AbsCriterion_init(lua_State *L);
extern void nn_SparseLinear_init(lua_State *L);

DLL_EXPORT int luaopen_libnn(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "nn");

  nn_Mean_init(L);
  nn_Min_init(L);
  nn_Max_init(L);
  nn_Sum_init(L);

  nn_Exp_init(L);
  nn_HardTanh_init(L);
  nn_LogSigmoid_init(L);
  nn_LogSoftMax_init(L);
  nn_Sigmoid_init(L);
  nn_SoftMax_init(L);
  nn_SoftPlus_init(L);
  nn_Tanh_init(L);

  nn_SpatialConvolution_init(L);
  nn_SpatialSubSampling_init(L);
  nn_TemporalConvolution_init(L);
  nn_TemporalSubSampling_init(L);
  nn_SparseLinear_init(L);
  nn_MSECriterion_init(L);
  nn_AbsCriterion_init(L);

  return 1;
}
