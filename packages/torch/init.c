#include "Storage.h"
#include "Tensor.h"
#include "TensorMath.h"

#include "File.h"
#include "DiskFile.h"
#include "MemoryFile.h"
#include "PipeFile.h"

#include "Timer.h"

extern void torch_utils_init(lua_State *L);

static lua_State *globalL;
static void luaTorchErrorHandlerFunction(const char *msg)
{
  luaL_error(globalL, msg);
}

static void luaTorchArgCheckHandlerFunction(int condition, int argNumber, const char *msg)
{
  luaL_argcheck(globalL, condition, argNumber, msg);
}

DLL_EXPORT int luaopen_libtorch(lua_State *L)
{
  globalL = L;
  THSetErrorHandler(luaTorchErrorHandlerFunction);
  THSetArgCheckHandler(luaTorchArgCheckHandlerFunction);

  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "torch");

  torch_utils_init(L);

  torch_ByteStorage_init(L);
  torch_CharStorage_init(L);
  torch_ShortStorage_init(L);
  torch_IntStorage_init(L);
  torch_LongStorage_init(L);
  torch_FloatStorage_init(L);
  torch_DoubleStorage_init(L);

  torch_ByteTensor_init(L);
  torch_CharTensor_init(L);
  torch_ShortTensor_init(L);
  torch_IntTensor_init(L);
  torch_LongTensor_init(L);
  torch_FloatTensor_init(L);
  torch_Tensor_init(L);
  torch_TensorMath_init(L);

  torch_Timer_init(L);
  torch_File_init(L);
  torch_DiskFile_init(L);
  torch_PipeFile_init(L);
  torch_MemoryFile_init(L);

  return 1;
}
