// -*- C++ -*-

#ifndef QTCORE_H
#define QTCORE_H

#include "lua.h"
#include "lauxlib.h"
#include "qtluaengine.h"
#include "qtluautils.h"


#ifdef LUA_BUILD_AS_DLL
# ifdef libqtcore_EXPORTS
#  define QTCORE_API __declspec(dllexport)
# else
#  define QTCORE_API __declspec(dllimport)
# endif
#else
# define QTCORE_API /**/
#endif

LUA_EXTERNC QTCORE_API int luaopen_libqtcore(lua_State *L);


#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


