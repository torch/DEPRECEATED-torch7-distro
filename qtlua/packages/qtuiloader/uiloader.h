// -*- C++ -*-

#ifndef UILOADER_H
#define UILOADER_H

#include "lua.h"
#include "lauxlib.h"
#include "qtluaengine.h"
#include "qtluautils.h"


#ifdef LUA_BUILD_AS_DLL
# ifdef libqtuiloader_EXPORTS
#  define QTUILOADER_API __declspec(dllexport)
# else
#  define QTUILOADER_API __declspec(dllimport)
# endif
#else
# define QTUILOADER_API /**/
#endif

LUA_EXTERNC QTUILOADER_API int luaopen_libqtuiloader(lua_State *L);


#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


