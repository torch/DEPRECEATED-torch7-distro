// -*- C++ -*-

#ifndef QTTORCH_H
#define QTTORCH_H

#ifdef LUA_NOT_CXX
#include "lua.hpp"
#else
#include "lua.h"
#include "lauxlib.h"
#endif

#include "qtluaengine.h"
#include "qtluautils.h"


#ifdef LUA_BUILD_AS_DLL
# ifdef libqttorch_EXPORTS
#  define QTTORCH_API __declspec(dllexport)
# else
#  define QTTORCH_API __declspec(dllimport)
# endif
#else
# define QTTORCH_API /**/
#endif

#ifndef LUA_NOT_CXX
LUA_EXTERNC
#endif
QTTORCH_API int luaopen_libqttorch(lua_State *L);


#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


