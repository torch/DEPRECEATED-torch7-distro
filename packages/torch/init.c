#include "general.h"

extern void torch_utils_init(lua_State *L);
extern void torch_File_init(lua_State *L);
extern void torch_File_init_storage_id(lua_State *L);
extern void torch_DiskFile_init(lua_State *L);
extern void torch_MemoryFile_init(lua_State *L);
extern void torch_PipeFile_init(lua_State *L);
extern void torch_Timer_init(lua_State *L);

extern void torch_ByteStorage_init(lua_State *L);
extern void torch_CharStorage_init(lua_State *L);
extern void torch_ShortStorage_init(lua_State *L);
extern void torch_IntStorage_init(lua_State *L);
extern void torch_LongStorage_init(lua_State *L);
extern void torch_FloatStorage_init(lua_State *L);
extern void torch_DoubleStorage_init(lua_State *L);

extern void torch_ByteTensor_init(lua_State *L);
extern void torch_CharTensor_init(lua_State *L);
extern void torch_ShortTensor_init(lua_State *L);
extern void torch_IntTensor_init(lua_State *L);
extern void torch_LongTensor_init(lua_State *L);
extern void torch_FloatTensor_init(lua_State *L);
extern void torch_DoubleTensor_init(lua_State *L);

/* extern void torch_ByteTensorMath_init(lua_State *L); */
/* extern void torch_CharTensorMath_init(lua_State *L); */
/* extern void torch_ShortTensorMath_init(lua_State *L); */
/* extern void torch_IntTensorMath_init(lua_State *L); */
/* extern void torch_LongTensorMath_init(lua_State *L); */
/* extern void torch_FloatTensorMath_init(lua_State *L); */
/* extern void torch_DoubleTensorMath_init(lua_State *L); */

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
  torch_File_init(L);

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
  torch_DoubleTensor_init(L);

  torch_File_init_storage_id(L);

/*   torch_ByteTensorMath_init(L); */
/*   torch_CharTensorMath_init(L); */
/*   torch_ShortTensorMath_init(L); */
/*   torch_IntTensorMath_init(L); */
/*   torch_LongTensorMath_init(L); */
/*   torch_FloatTensorMath_init(L); */
/*   torch_DoubleTensorMath_init(L); */

  torch_Timer_init(L);
  torch_DiskFile_init(L);
  torch_PipeFile_init(L);
  torch_MemoryFile_init(L);

  return 1;
}
