// -*- C++ -*-

#ifndef QLUAAPPLICATION_H
#define QLUAAPPLICATION_H

#include "qluaconf.h"

#include <QApplication>
#include <QEvent>
#include <QString>
#include <QStringList>
#include <QVariant>

#define QLUA_ORG   "QLua"
#define LUA_DOMAIN "lua.org"

class QtLuaEngine;

class QLuaConsole;

class QLUAAPI QLuaApplication : public QApplication
{
  Q_OBJECT
  Q_PROPERTY(QString aboutMessage READ aboutMessage WRITE setAboutMessage)

public:
  ~QLuaApplication();
  QLuaApplication(int &argc, char **argv, 
                  bool guiEnabled=true, bool oneThread=false);
  int main(int argc, char **argv);
  static QLuaApplication *instance();
  static QLuaConsole *console();
  static QtLuaEngine *engine();
  bool isAcceptingCommands() const;
  bool isClosingDown() const;
  QString aboutMessage() const;
  Q_INVOKABLE double timeForLastCommand() const;
  Q_INVOKABLE QVariant readSettings(QString key);
  Q_INVOKABLE void writeSettings(QString key, QVariant value);
  Q_INVOKABLE QStringList filesToOpen();
  Q_INVOKABLE bool runsWithoutGraphics() const;
  
public slots:
  void setupConsoleOutput();
  void setAboutMessage(QString msg);
  bool runCommand(QByteArray cmd);
  void restart(bool redoCmdLine=true);
  bool close();
  void about();

protected:
  bool event(QEvent *e);
  bool notify(QObject *receiver, QEvent *event);
  
signals:
  void newEngine(QtLuaEngine *engine);
  void acceptingCommands(bool);
  void luaCommandEcho(QByteArray ps1, QByteArray ps2, QByteArray cmd);
  void luaConsoleOutput(QByteArray out);
  void anyoneHandlesConsoleOutput(bool &pbool);
  void fileOpenEvent();
  void windowShown(QWidget *window);
  
private:
  struct Private;
  Private *d;
};

// Silly logic for mac bugs
#ifdef Q_WS_MAC
# ifndef QT_MAC_USE_COCOA
#   define QT_MAC_USE_NATIVE_MENUBAR 1
# else
#  define QT_MAC_STEAL_STATUSBAR 1
#  if QT_VERSION >= 0xFFFFFF // Not fixed in 40503 yet
#   define QT_MAC_USE_NATIVE_MENUBAR 1
#  endif
# endif
#endif

#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ( "\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" )
   End:
   ------------------------------------------------------------- */

