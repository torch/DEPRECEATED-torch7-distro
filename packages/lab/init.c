#include "luaT.h"

extern void lab_init(lua_State *L);

extern void lab_Byteinit(lua_State *L);
extern void lab_Charinit(lua_State *L);
extern void lab_Shortinit(lua_State *L);
extern void lab_Intinit(lua_State *L);
extern void lab_Longinit(lua_State *L);
extern void lab_Floatinit(lua_State *L);
extern void lab_Doubleinit(lua_State *L);

extern void lab_Byteconv_init(lua_State *L);
extern void lab_Charconv_init(lua_State *L);
extern void lab_Shortconv_init(lua_State *L);
extern void lab_Intconv_init(lua_State *L);
extern void lab_Longconv_init(lua_State *L);
extern void lab_Floatconv_init(lua_State *L);
extern void lab_Doubleconv_init(lua_State *L);

extern void lab_Floatlapack_init(lua_State *L);
extern void lab_Doublelapack_init(lua_State *L);

extern void lab_utils_init(lua_State *L);

DLL_EXPORT int luaopen_liblab(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "lab");

  lab_init(L);

  lab_Byteconv_init(L);
  lab_Charconv_init(L);
  lab_Shortconv_init(L);
  lab_Intconv_init(L);
  lab_Longconv_init(L);
  lab_Floatconv_init(L);
  lab_Doubleconv_init(L);

  lab_Floatlapack_init(L);
  lab_Doublelapack_init(L);

  lab_utils_init(L);

  return 1;
}
