// -*- C++ -*-

#ifndef QTLUAGUI_H
#define QTLUAGUI_H

#include "qtgui.h"

#include <QAction>
#include <QCoreApplication>
#include <QObject>
#include <QPointer>



class QTGUI_API QtLuaAction : public QAction 
{
  Q_OBJECT
  Q_PROPERTY(bool enabled READ isEnabled WRITE setEnabled)
  Q_PROPERTY(bool autoDisable READ autoDisable WRITE setAutoDisable)
public:
  QtLuaAction(QtLuaEngine *e = 0, QObject *p = 0);
  bool isEnabled() const { return enabled; }
  bool autoDisable() const { return !override; }
public slots:
  void stateChanged();
  void setEngine(QtLuaEngine *e);
  void setEnabled(bool b) { enabled=b; stateChanged(); }  
  void setDisabled(bool b) { enabled=!b; stateChanged(); }
  void setAutoDisable(bool b) { override=!b; stateChanged(); }
private:
  bool enabled;
  bool override;
  QPointer<QtLuaEngine> engine;
};




#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*" "qreal")
   End:
   ------------------------------------------------------------- */


