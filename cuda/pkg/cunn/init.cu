#include "luaT.h"
#include "THC.h"

#include <thrust/transform.h>

const void *torch_CudaTensor_id = NULL;

#include "HardTanh.cu"

DLL_EXPORT TH_API int luaopen_libcunn(lua_State *L)
{
  lua_newtable(L);

  torch_CudaTensor_id = luaT_checktypename2id(L, "torch.CudaTensor");

  cunn_HardTanh_init(L);

  return 1;
}
