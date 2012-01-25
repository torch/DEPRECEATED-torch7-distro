/* -*- C++ -*- */


#include "qtluagui.h"



QtLuaAction::QtLuaAction(QtLuaEngine *e, QObject *p)
  : QAction(p), enabled(true), override(false), engine(0)
{
  setEngine(e);
  QCoreApplication *app = QCoreApplication::instance();
  if (app && app->inherits("QLuaApplication"))
    connect(app, SIGNAL(newEngine(QtLuaEngine*)), 
            this, SLOT(setEngine(QtLuaEngine*)) );
}

void 
QtLuaAction::stateChanged()
{
  bool ready = override || (engine && engine->runSignalHandlers());
  QAction::setEnabled(ready && enabled);
}


void 
QtLuaAction::setEngine(QtLuaEngine *e)
{
  if (engine)
    disconnect(engine, 0, this, 0);
  if ((engine = e))
    connect(engine, SIGNAL(stateChanged(int)),
            this, SLOT(stateChanged()) );
  stateChanged();
}





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*" "qreal")
   End:
   ------------------------------------------------------------- */
