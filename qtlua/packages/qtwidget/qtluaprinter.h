// -*- C++ -*-

#ifndef QTLUAPRINTER_H
#define QTLUAPRINTER_H

#ifdef LUA_NOT_CXX
#include "lua.hpp"
#else
#include "lua.h"
#include "lauxlib.h"
#endif

#include "qtluaengine.h"
#include "qtluautils.h"

#include "qtwidget.h"

#include <QObject>
#include <QPrinter>
#include <QRect>
#include <QSizeF>
#include <QString>



class QTWIDGET_API QtLuaPrinter : public QObject, public QPrinter
{
  Q_OBJECT
  Q_PROPERTY(bool colorMode READ colorMode WRITE setColorMode)
  Q_PROPERTY(bool collateCopies READ collateCopies WRITE setCollateCopies)
  Q_PROPERTY(QString creator READ creator WRITE setCreator)
  Q_PROPERTY(QString docName READ docName WRITE setDocName)
  Q_PROPERTY(bool doubleSidedPrinting READ doubleSidedPrinting 
             WRITE setDoubleSidedPrinting)
  Q_PROPERTY(bool fontEmbeddingEnabled READ fontEmbeddingEnabled 
             WRITE setFontEmbeddingEnabled)
  Q_PROPERTY(int fromPage READ fromPage)
  Q_PROPERTY(bool fullPage READ fullPage WRITE setFullPage)
  Q_PROPERTY(bool landscape READ landscape WRITE setLandscape)
  Q_PROPERTY(int numCopies READ numCopies WRITE setNumCopies)
  Q_PROPERTY(QString outputFileName READ outputFileName WRITE setOutputFileName)
  Q_PROPERTY(QString outputFormat READ outputFormat WRITE setOutputFormat)
  Q_PROPERTY(QString pageSize READ pageSize WRITE setPageSize)
  Q_PROPERTY(QString printerName READ printerName WRITE setPrinterName)
  Q_PROPERTY(QString printProgram READ printProgram WRITE setPrintProgram)
  Q_PROPERTY(int resolution READ resolution WRITE setResolution)
  Q_PROPERTY(int toPage READ toPage)
  Q_PROPERTY(QRect paperRect READ paperRect)
  Q_PROPERTY(QRect pageRect READ pageRect)
  Q_PROPERTY(QSizeF paperSize READ paperSize WRITE setPaperSize)
  Q_PROPERTY(QString printerState READ printerState)
  Q_ENUMS(PrinterState)

public:
 ~QtLuaPrinter();
  QtLuaPrinter(PrinterMode mode, QObject *parent=0)
    : QObject(parent), QPrinter(mode), custom(false) {}
  QtLuaPrinter(QObject *parent=0)
    : QObject(parent), QPrinter(), custom(false) {}

  Q_INVOKABLE QPrinter* printer() { return static_cast<QPrinter*>(this);}
  Q_INVOKABLE QPaintDevice* device() { return static_cast<QPaintDevice*>(this);}
  Q_INVOKABLE void setFromTo(int f, int t) { QPrinter::setFromTo(f, t); }
  Q_INVOKABLE bool newPage() { return QPrinter::newPage(); }
  Q_INVOKABLE bool abort() { return QPrinter::abort(); }
  Q_INVOKABLE bool setup(QWidget *parent=0);

  bool colorMode() const { return QPrinter::colorMode()==Color;}
  void setColorMode(bool b) { QPrinter::setColorMode(b?Color:GrayScale);}
  bool landscape() const { return orientation()==Landscape;}
  void setLandscape(bool b) { setOrientation(b?Landscape:Portrait);}
  bool lastPageFirst() const { return pageOrder()==LastPageFirst;}
  void setLastPageFirst(bool b) { setPageOrder(b?LastPageFirst:FirstPageFirst);}
  QString pageSize() const;
  void setPageSize(QString s);
  QString outputFormat() const;
  void setOutputFormat(QString s);
  QString printerState() const;
  QSizeF paperSize() const;
  void setPaperSize(QSizeF s);
private:
  QSizeF papSize;
  bool custom;
signals:
  void closing(QObject*);
};




#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*" "qreal")
   End:
   ------------------------------------------------------------- */


