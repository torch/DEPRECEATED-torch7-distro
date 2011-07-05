// -*- C++ -*-

#ifndef QTGUI_H
#define QTGUI_H

#include "lua.h"
#include "lauxlib.h"
#include "qtluaengine.h"
#include "qtluautils.h"


#ifdef LUA_BUILD_AS_DLL
# ifdef libqtgui_EXPORTS
#  define QTGUI_API __declspec(dllexport)
# else
#  define QTGUI_API __declspec(dllimport)
# endif
#else
# define QTGUI_API /**/
#endif

LUA_EXTERNC QTGUI_API int luaopen_libqtgui(lua_State *L);


#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


