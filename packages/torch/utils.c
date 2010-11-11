#include "general.h"

void torch_utils_init(lua_State *L)
{
  /* utility functions */
  lua_pushcfunction(L, luaT_lua_factory);
  lua_setfield(L, -2, "factory");
  lua_pushcfunction(L, luaT_lua_typename);
  lua_setfield(L, -2, "typename");
  lua_pushcfunction(L, luaT_lua_isequal);
  lua_setfield(L, -2, "isequal");
  lua_pushcfunction(L, luaT_lua_getenv);
  lua_setfield(L, -2, "getenv");
  lua_pushcfunction(L, luaT_lua_setenv);
  lua_setfield(L, -2, "setenv");
  lua_pushcfunction(L, luaT_lua_newmetatable);
  lua_setfield(L, -2, "newmetatable");
  lua_pushcfunction(L, luaT_lua_getmetatable);
  lua_setfield(L, -2, "getmetatable");
  lua_pushcfunction(L, luaT_lua_setmetatable);
  lua_setfield(L, -2, "setmetatable");
  lua_pushcfunction(L, luaT_lua_version);
  lua_setfield(L, -2, "version");
  lua_pushcfunction(L, luaT_lua_pointer);
  lua_setfield(L, -2, "pointer");
}
