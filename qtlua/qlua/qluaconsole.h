// -*- C++ -*-

#ifndef QLUACONSOLE_H
#define QLUACONSOLE_H

#include "qluaconf.h"

#include <QByteArray>
#include <QObject>
#include <QString>

class QtLuaEngine;

class QLUAAPI QLuaConsole : public QObject
{
  Q_OBJECT
  Q_PROPERTY(bool captureOutput 
             READ captureOutput WRITE setCaptureOutput)
  Q_PROPERTY(bool printCapturedOutput
             READ printCapturedOutput WRITE setPrintCapturedOutput)
public:
  QLuaConsole(QObject *parent);
  bool captureOutput() const;
  bool printCapturedOutput() const;
  
  typedef void (*CtrlCHandler)(int);
  static const CtrlCHandler ctrlCDefault;
  static const CtrlCHandler ctrlCIgnore;
  static const CtrlCHandler ctrlCBreak;
  CtrlCHandler setCtrlCHandler(CtrlCHandler handler);
  
public slots:
  void setCaptureOutput(bool);
  void setPrintCapturedOutput(bool);
  void setQtLuaEngine(QtLuaEngine *engine, bool breakStopsLua=false);
  void readLine(QByteArray prompt);
  void abortReadLine();
  void redisplayReadLine();
  void addToHistory(QByteArray line);
  void drainOutput();

signals:
  void consoleOutput(QByteArray output);
  void ttyInput(QByteArray input);
  void ttyEndOfFile();
  void ttyBreak();

public:
  struct Private;
private:
  Private *d;
};



#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ( "\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" )
   End:
   ------------------------------------------------------------- */

