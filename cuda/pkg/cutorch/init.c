#include "luaT.h"

DLL_EXPORT int luaopen_libcutorch(lua_State *L)
{
  torch_CudaStorage_init(L);

  return 1;
}
