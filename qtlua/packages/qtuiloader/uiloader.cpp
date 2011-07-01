// -*- C++ -*-


#include "uiloader.h"
#include "lualib.h"

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QLayout>
#include <QObject>
#include <QObject>
#include <QStringList>
#include <QUiLoader>
#include <QVariant>
#include <QWidget>


#include <stdio.h>

// ====================================


static int 
qtuiloader_new(lua_State *L)
{
  if (QApplication::type() == QApplication::Tty)
    luaL_error(L, "Graphics have been disabled (running with -nographics)");
  QObject *parent = luaQ_optqobject<QObject>(L, 1, 0);
  QUiLoader *q = new QUiLoader(parent);
  luaQ_pushqt(L, q, !parent);
  return 1;
}


static int
qtuiloader_addPluginPath(lua_State *L)
{
  QUiLoader *loader = luaQ_checkqobject<QUiLoader>(L, 1);
  const char *path = luaL_checkstring(L, 2);
  loader->addPluginPath(QFile::decodeName(path));
  return 0;
}


static int
qtuiloader_availableWidgets(lua_State *L)
{
  QUiLoader *loader = luaQ_checkqobject<QUiLoader>(L, 1);
  luaQ_pushqt(L, QVariant(loader->availableWidgets()));
  return 1;
}


static int
qtuiloader_clearPluginPaths(lua_State *L)
{
  QUiLoader *loader = luaQ_checkqobject<QUiLoader>(L, 1);
  loader->clearPluginPaths();
  return 0;
}


static int
qtuiloader_createAction(lua_State *L)
{
  QUiLoader *loader = luaQ_checkqobject<QUiLoader>(L, 1);
  QObject *parent = luaQ_optqobject<QObject>(L, 2, 0);
  QString name = luaQ_optqvariant<QString>(L, 3, QString());
  luaQ_pushqt(L, loader->createAction(parent, name), !parent);
  return 1;
}


static int
qtuiloader_createActionGroup(lua_State *L)
{
  QUiLoader *loader = luaQ_checkqobject<QUiLoader>(L, 1);
  QObject *parent = luaQ_optqobject<QObject>(L, 2, 0);
  QString name = luaQ_optqvariant<QString>(L, 3, QString());
  luaQ_pushqt(L, loader->createActionGroup(parent, name), !parent);
  return 1;
}


static int
qtuiloader_createLayout(lua_State *L)
{
  QUiLoader *loader = luaQ_checkqobject<QUiLoader>(L, 1);
  QString classname = luaQ_checkqvariant<QString>(L, 2);
  QObject *parent = luaQ_optqobject<QObject>(L, 3, 0);
  QString name = luaQ_optqvariant<QString>(L, 4, QString());
  luaQ_pushqt(L, loader->createLayout(classname, parent, name), !parent);
  return 1;
}


static int
qtuiloader_createWidget(lua_State *L)
{
  QUiLoader *loader = luaQ_checkqobject<QUiLoader>(L, 1);
  QString classname = luaQ_checkqvariant<QString>(L, 2);
  QWidget *parent = luaQ_optqobject<QWidget>(L, 3, 0);
  QString name = luaQ_optqvariant<QString>(L, 4, QString());
  QWidget *widget = loader->createWidget(classname, parent, name);
  luaQ_pushqt(L, widget, !parent);
  return 1;
}


static int
qtuiloader_load(lua_State *L)
{
  // this
  QUiLoader *loader = luaQ_checkqobject<QUiLoader>(L, 1);
  // file
  QFile afile;
  QIODevice *file = qobject_cast<QIODevice*>(luaQ_toqobject(L, 2));
  if (!file && lua_isstring(L, 2))
    {
      file = &afile;
      const char *fn = lua_tostring(L, 2);
      afile.setFileName(QFile::decodeName(fn));
      if (! afile.open(QIODevice::ReadOnly))
        luaL_error(L,"cannot open file '%s' for reading (%s)", 
                   fn, afile.errorString().toLocal8Bit().constData() );
    }
  else if (!file)
    {
      file = &afile;
      void *udata = luaL_checkudata(L, 2, LUA_FILEHANDLE);
      if (! afile.open(*(FILE**)udata, QIODevice::ReadOnly))
        luaL_error(L,"cannot use stream for reading (%s)", 
                   afile.errorString().toLocal8Bit().constData() );
    }
  // parent
  QWidget *parent = luaQ_optqobject<QWidget>(L, 3);
  // load
  QWidget *w = loader->load(file, parent);
  luaQ_pushqt(L, w, !parent);
  return 1;
}


static int
qtuiloader_pluginPaths(lua_State *L)
{
  QUiLoader *loader = luaQ_checkqobject<QUiLoader>(L, 1);
  luaQ_pushqt(L, QVariant(loader->pluginPaths()));
  return 1;
}


static int
qtuiloader_setWorkingDirectory(lua_State *L)
{
  QUiLoader *loader = luaQ_checkqobject<QUiLoader>(L, 1);
  const char *dir = luaL_checkstring(L, 2);
  loader->setWorkingDirectory(QDir(QFile::decodeName(dir)));
  return 0;
}


static int
qtuiloader_workingDirectory(lua_State *L)
{
  QUiLoader *loader = luaQ_checkqobject<QUiLoader>(L, 1);
  QString path = loader->workingDirectory().path();
  lua_pushstring(L, QFile::encodeName(path).constData());
  return 1;
}




// =====================================

static const luaL_Reg qtuiloader_lib[] = {
  {"new", qtuiloader_new},
  {"addPluginPath", qtuiloader_addPluginPath},
  {"availableWidgets", qtuiloader_availableWidgets},
  {"clearPluginPaths", qtuiloader_clearPluginPaths},
  {"createAction", qtuiloader_createAction},
  {"createActionGroup", qtuiloader_createActionGroup},
  {"createLayout", qtuiloader_createLayout},
  {"createWidget", qtuiloader_createWidget},
  {"load", qtuiloader_load},
  {"pluginPaths", qtuiloader_pluginPaths},
  {"setWorkingDirectory", qtuiloader_setWorkingDirectory},
  {"workingDirectory", qtuiloader_workingDirectory},
  {NULL, NULL}
};


LUA_EXTERNC QTUILOADER_API int 
luaopen_libqtuiloader(lua_State *L)
{
  // load module 'qt'
  if (luaL_dostring(L, "require 'qt'"))
    lua_error(L);
  // enrichs class QUiLoader.
  luaQ_pushmeta(L, &QUiLoader::staticMetaObject);
  luaQ_getfield(L, -1, "__metatable");
  luaQ_register(L, qtuiloader_lib, QCoreApplication::instance());
  return 0;
}



/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


