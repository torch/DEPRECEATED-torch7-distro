// -*- C++ -*-

#ifndef QTLUALISTENER_H
#define QTLUALISTENER_H

#include "lua.h"
#include "lauxlib.h"
#include "qtluaengine.h"
#include "qtluautils.h"

#include "qtwidget.h"


#include <QEvent>
#include <QWidget>


class QTWIDGET_API QtLuaListener : public QObject
{
  Q_OBJECT
public:
  QtLuaListener(QWidget *w);

signals:
  void sigClose();
  void sigResize(int w, int h);
  void sigKeyPress(QString text, QByteArray key, QByteArray modifiers);
  void sigKeyRelease(QString text, QByteArray key, QByteArray modifiers);
  void sigMousePress(int x, int y, QByteArray button, QByteArray m, QByteArray b);
  void sigMouseRelease(int x, int y, QByteArray button, QByteArray m, QByteArray b);
  void sigMouseDoubleClick(int x, int y, QByteArray button, QByteArray m, QByteArray b);
  void sigMouseMove(int x, int y, QByteArray m, QByteArray b);
  void sigEnter(bool enter);
  void sigFocus(bool focus);
  void sigShow(bool show);
  void sigPaint();
  
protected:
  bool eventFilter(QObject *object, QEvent *event);
private:
  QWidget *w;
};


#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


