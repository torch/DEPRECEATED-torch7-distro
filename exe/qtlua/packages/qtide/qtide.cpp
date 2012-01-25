/* -*- C++ -*- */


#include "qtide.h"
#include "qluatextedit.h"
#include "qluamainwindow.h"
#include "qluabrowser.h"
#include "qluaeditor.h"
#include "qluaide.h"
#include "qluasdimain.h"
#include "qluamdimain.h"
#include "qtluaengine.h"
#include "qtluautils.h"

#include <QApplication>
#include <QMessageBox>
#include <QPrinter>


#include <stdlib.h>


// ========================================
// QLUAEDITOR


int 
qluaide_new(lua_State *L)
{
  luaQ_pushqt(L, QLuaIde::instance());
  return 1;
}


static int
no_methodcall(lua_State *L)
{
  luaL_error(L, "This class prevents lua to call this method");
  return 0;
}


static luaL_Reg qluaide_lib[] = {
  { "new", qluaide_new },
  { "deleteLater", no_methodcall },
  { 0, 0 }
};


static int qluaide_hook(lua_State *L) 
{
  lua_getfield(L, -1, "__metatable");
  luaQ_register(L, qluaide_lib, QCoreApplication::instance());
  return 0;
}





// ========================================
// REGISTER




LUA_EXTERNC QTIDE_API int
luaopen_libqtide(lua_State *L)
{ 
  // load module 'qt'
  if (luaL_dostring(L, "require 'qt'"))
    lua_error(L);
  if (QApplication::type() == QApplication::Tty)
    luaL_error(L, "Graphics have been disabled (running with -nographics)");
  // register metatypes
  qRegisterMetaType<QPrinter*>("QPrinter*");
  // register classes
  QtLuaEngine::registerMetaObject(&QLuaBrowser::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QLuaTextEdit::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QLuaTextEditMode::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QLuaConsoleWidget::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QLuaEditor::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QLuaSdiMain::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QLuaMdiMain::staticMetaObject);
#if HAVE_QTWEBKIT
  QtLuaEngine::registerMetaObject(&QWebFrame::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QWebPage::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QWebView::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QLuaBrowser::staticMetaObject);
#endif
  // class 'qluaide'
  lua_pushcfunction(L, qluaide_hook);
  luaQ_pushmeta(L, &QLuaIde::staticMetaObject);
  lua_call(L, 1, 0);
  // return
  return 1;
}




/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


