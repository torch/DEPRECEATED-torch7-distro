// -*- C -*-

#ifndef QTLUAUTILS_H
#define QTLUAUTILS_H

#ifdef LUA_NOT_CXX
#include "lua.hpp"
#else
#include "lua.h"
#include "lauxlib.h"
#endif

#include "qtluaconf.h"

#ifdef WIN32
# ifdef libqtlua_EXPORTS
#  define QTLUAAPI __declspec(dllexport)
# else
#  define QTLUAAPI __declspec(dllimport)
# endif
#else
# define QTLUAAPI
#endif


#ifdef __cplusplus
# define QTLUA_EXTERNC extern "C"
#else
# define QTLUA_EXTERNC extern
#endif

QTLUA_EXTERNC QTLUAAPI void luaQ_getfield(lua_State *L, int index, const char *name);
QTLUA_EXTERNC QTLUAAPI int  luaQ_tracebackskip(lua_State *L, int skip);
QTLUA_EXTERNC QTLUAAPI int  luaQ_traceback(lua_State *L);
QTLUA_EXTERNC QTLUAAPI int  luaQ_complete(lua_State *L);
QTLUA_EXTERNC QTLUAAPI int  luaQ_print(lua_State *L, int nr);
QTLUA_EXTERNC QTLUAAPI int  luaQ_pcall(lua_State *L, int na, int nr, int eh, int oh);
QTLUA_EXTERNC QTLUAAPI void luaQ_doevents(lua_State *L);
QTLUA_EXTERNC QTLUAAPI int  luaopen_qt(lua_State *L);

#endif




/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ( "\\sw+_t" "lua_[A-Z]\\sw*[a-z]\\sw*" )
   c-font-lock-extra-types: ( "\\sw+_t" "lua_[A-Z]\\sw*[a-z]\\sw*" )
   End:
   ------------------------------------------------------------- */
