// -*- C++ -*-

#ifndef QTLUASVGGENERATOR_H
#define QTLUASVGGENERATOR_H

#ifdef LUA_NOT_CXX
#include "lua.hpp"
#else
#include "lua.h"
#include "lauxlib.h"
#endif

#include "qtluaengine.h"
#include "qtluautils.h"

#include "qtsvg.h"

#include <QByteArray>
#include <QObject>
#include <QRect>
#include <QSize>
#include <QString>
#include <QSvgGenerator>

class QTSVG_API QtLuaSvgGenerator : public QObject, public QSvgGenerator
{
  Q_OBJECT
  Q_PROPERTY(QString description READ description WRITE setDescription)
  Q_PROPERTY(QSize size READ size WRITE setSize)
  Q_PROPERTY(QString title READ title WRITE setTitle)
  Q_PROPERTY(int resolution READ resolution WRITE setResolution)
 public:
  ~QtLuaSvgGenerator();
  QtLuaSvgGenerator(QObject *parent=0);
  QtLuaSvgGenerator(QString fileName, QObject *parent=0);
  Q_INVOKABLE QPaintDevice* device() { return this; }
  Q_INVOKABLE QByteArray data();
  QString description() const;
  QSize size() const;
  QString title() const;
  int resolution() const;
 public slots:
  void setDescription(QString s);
  void setSize(QSize s);
  void setTitle(QString s);
  void setResolution(int r);
 public:  
  struct Private;
 private:
  Private *d;
 signals:
  void closing(QObject*);
};




#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*" "qreal")
   End:
   ------------------------------------------------------------- */


