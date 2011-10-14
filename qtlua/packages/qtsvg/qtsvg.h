// -*- C++ -*-

#ifndef QTSVG_H
#define QTSVG_H

#ifdef LUA_NOT_CXX
#include "lua.hpp"
#else
#include "lua.h"
#include "lauxlib.h"
#endif

#include "qtluaengine.h"
#include "qtluautils.h"


#ifdef LUA_BUILD_AS_DLL
# ifdef libqtsvg_EXPORTS
#  define QTSVG_API __declspec(dllexport)
# else
#  define QTSVG_API __declspec(dllimport)
# endif
#else
# define QTSVG_API /**/
#endif

#ifndef LUA_NOT_CXX
LUA_EXTERNC
#endif
QTSVG_API int luaopen_libqtsvg(lua_State *L);


#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


