// -*- C++ -*-

#ifndef QTWIDGET_H
#define QTWIDGET_H

#ifdef LUA_NOT_CXX
#include "lua.hpp"
#else
#include "lua.h"
#include "lauxlib.h"
#endif

#include "qtluaengine.h"
#include "qtluautils.h"


#ifdef LUA_BUILD_AS_DLL
# ifdef libqtwidget_EXPORTS
#  define QTWIDGET_API __declspec(dllexport)
# else
#  define QTWIDGET_API __declspec(dllimport)
# endif
#else
# define QTWIDGET_API /**/
#endif


#ifndef LUA_NOT_CXX
LUA_EXTERNC
#endif
QTWIDGET_API int luaopen_libqtwidget(lua_State *L);


#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


