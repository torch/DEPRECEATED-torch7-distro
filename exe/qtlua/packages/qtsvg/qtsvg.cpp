// -*- C++ -*-

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>

#include <QCoreApplication>
#include <QMetaMethod>
#include <QMetaObject>
#include <QMetaType>
#include <QObject>
#include <QPaintDevice>
#include <QPainter>
#include <QSvgGenerator>
#include <QSvgRenderer>
#include <QSvgWidget>
#include <QVariant>

#include "qtsvg.h"
#include "qtluasvggenerator.h"


Q_DECLARE_METATYPE(QPainter*)
Q_DECLARE_METATYPE(QPaintDevice*)


// ====================================
// QTLUASVGGENERATOR

static int 
qtluasvggenerator_new(lua_State *L)
{
  QVariant v = luaQ_toqvariant(L, 1, QMetaType::QString);
  if (v.type() == QVariant::String)
    {
      QObject *p = luaQ_optqobject<QObject>(L, 2);
      luaQ_pushqt(L, new QtLuaSvgGenerator(v.toString(), p), !p);
    }
  else
    {
      QObject *p = luaQ_optqobject<QObject>(L, 1);
      luaQ_pushqt(L, new QtLuaSvgGenerator(p), !p);
    }
  return 1;
}

static struct luaL_Reg qtluasvggenerator_lib[] = {
  {"new", qtluasvggenerator_new},
  {0,0}
};

static int 
qtluasvggenerator_hook(lua_State *L)
{
  lua_getfield(L, -1, "__metatable");
  luaQ_register(L, qtluasvggenerator_lib, QCoreApplication::instance());
  return 0;
}


// ====================================
// QSVGRENDERER

static int 
qsvgrenderer_new(lua_State *L)
{
  QVariant v = luaQ_toqvariant(L, 1, QMetaType::QString);
  if (v.type() == QVariant::String)
    {
      QObject *p = luaQ_optqobject<QObject>(L, 2);
      luaQ_pushqt(L, new QSvgRenderer(v.toString(), p), !p);
    }
  else
    {
      QObject *p = luaQ_optqobject<QObject>(L, 1);
      luaQ_pushqt(L, new QSvgRenderer(p), !p);
    }
  return 1;
}

static struct luaL_Reg qsvgrenderer_lib[] = {
  {"new", qsvgrenderer_new},
  {0,0}
};

static int 
qsvgrenderer_hook(lua_State *L)
{
  lua_getfield(L, -1, "__metatable");
  luaQ_register(L, qsvgrenderer_lib, QCoreApplication::instance());
  return 0;
}



// ====================================
// QSVGWIDGET


static int 
qsvgwidget_new(lua_State *L)
{
  QVariant v = luaQ_toqvariant(L, 1, QMetaType::QString);
  if (v.type() == QVariant::String)
    {
      QWidget *p = luaQ_optqobject<QWidget>(L, 2);
      luaQ_pushqt(L, new QSvgWidget(v.toString(), p), !p);
    }
  else
    {
      QWidget *p = luaQ_optqobject<QWidget>(L, 1);
      luaQ_pushqt(L, new QSvgWidget(p), !p);
    }
  return 1;
}

static int 
qsvgwidget_renderer(lua_State *L)
{
  QSvgWidget *w = luaQ_checkqobject<QSvgWidget>(L, 1);
  luaQ_pushqt(L, w->renderer());
  return 1;
}

static struct luaL_Reg qsvgwidget_lib[] = {
  {"new", qsvgwidget_new},
  {"renderer", qsvgwidget_renderer},
  {0,0}
};

static int 
qsvgwidget_hook(lua_State *L)
{
  lua_getfield(L, -1, "__metatable");
  luaQ_register(L, qsvgwidget_lib, QCoreApplication::instance());
  return 0;
}


// ====================================


LUA_EXTERNC QTSVG_API int
luaopen_libqtsvg(lua_State *L)
{
  // load module 'qt'
  if (luaL_dostring(L, "require 'qt'"))
    lua_error(L);

  // hooks for objects
#define HOOK_QOBJECT(T, t) \
     lua_pushcfunction(L, t ## _hook);\
     luaQ_pushmeta(L, &T::staticMetaObject);\
     lua_call(L, 1, 0);

  HOOK_QOBJECT(QtLuaSvgGenerator, qtluasvggenerator)
  HOOK_QOBJECT(QSvgRenderer, qsvgrenderer)
  HOOK_QOBJECT(QSvgWidget, qsvgwidget)

  return 0;
}



/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


