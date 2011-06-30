// -*- C++ -*-

#ifndef QTIDE_H
#define QTIDE_H

#include "lua.h"
#include "lauxlib.h"
#include "qtluaengine.h"
#include "qtluautils.h"


#ifdef LUA_BUILD_AS_DLL
# ifdef libqtide_EXPORTS
#  define QTIDE_API __declspec(dllexport)
# else
#  define QTIDE_API __declspec(dllimport)
# endif
#else
# define QTIDE_API /**/
#endif

LUA_EXTERNC QTIDE_API int luaopen_libqtide(lua_State *L);

#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */

