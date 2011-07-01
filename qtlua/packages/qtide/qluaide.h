// -*- C++ -*-

#ifndef QLUAIDE_H
#define QLUAIDE_H

#include "qtide.h"

#include <QAction>
#include <QByteArray>
#include <QFile>
#include <QList>
#include <QMenuBar>
#include <QMessageBox>
#include <QObject>
#include <QObjectList>
#include <QSize>
#include <QString>
#include <QUrl>
#include <QVariant>
#include <QWidget>

class QLuaMainWindow;
class QLuaEditor;
class QLuaInspector;
class QLuaSdiMain;
class QLuaMdiMain;
class QLuaBrowser;

// Text editor widget

class QTIDE_API QLuaIde : public QObject
{
  Q_OBJECT
  Q_PROPERTY(bool editOnError READ editOnError WRITE setEditOnError)
  Q_PROPERTY(bool mdiDefault READ mdiDefault)

public:
  static QLuaIde *instance();
  static QString fileDialogFilters();
  static QString htmlFilesFilter();
  static QString allFilesFilter();
                                             
  bool editOnError() const;
  bool mdiDefault() const;
  
  Q_INVOKABLE QObjectList windows() const;
  Q_INVOKABLE QStringList windowNames() const;
  Q_INVOKABLE QStringList recentFiles() const;
  Q_INVOKABLE QWidget* previousWindow() const;
  Q_INVOKABLE QWidget* activeWindow() const;
  Q_INVOKABLE QLuaSdiMain *sdiMain() const;
  Q_INVOKABLE QLuaMdiMain *mdiMain() const;
  Q_INVOKABLE QMenuBar *globalMenuBar() const;
  Q_INVOKABLE QMenuBar *defaultMenuBar(QMenuBar *menu);
  Q_INVOKABLE QAction *hasAction(QByteArray name);
  Q_INVOKABLE QAction *stdAction(QByteArray name);

public slots:
  QLuaEditor    *editor(QString fname = QString());
  QLuaBrowser   *browser(QUrl url = QUrl());
  QLuaBrowser   *browser(QString url);
  QLuaInspector *inspector(); 
  QLuaSdiMain   *createSdiMain();
  QLuaMdiMain   *createMdiMain();
  void setEditOnError(bool b);
  void addRecentFile(QString fname);
  void clearRecentFiles();
  void loadRecentFiles();
  void saveRecentFiles();
  void activateWidget(QWidget *w);
  void activateConsole(QWidget *returnTo=0);
  void loadWindowGeometry(QWidget *w);
  void saveWindowGeometry(QWidget *w);
  void updateActions();
  bool openFile(QString fileName, bool inOther=false, QWidget *window=0);
  bool newDocument(QWidget *window=0);
  bool luaExecute(QByteArray cmd);
  bool luaRestart(QByteArray cmd);
  bool quit(QWidget *r);

  int messageBox(QString t, QString m, 
                 QMessageBox::StandardButtons buttons,
                 QMessageBox::StandardButton def = QMessageBox::NoButton,
                 QMessageBox::Icon icon = QMessageBox::Warning);
  
  QByteArray messageBox(QString t, QString m, 
                        QByteArray buttons = "Ok",
                        QByteArray def = "NoButton",
                        QByteArray icon = "Warning");
  
  void doNew();
  void doOpen();
  bool doClose();
  bool doQuit();
  void doReturnToConsole();
  void doReturnToPrevious();
  void doLuaStop();
  void doLuaPause();
  void doPreferences();
  void doHelp();

signals:
  void windowsChanged();
  void prefsRequested(QWidget *window);
  void helpRequested(QWidget *window);

public:
  class Private;
protected:
  QLuaIde();
  QAction *newAction(QString title);
  QMenu *newMenu(QString title);
private:
  Private *d;
};




// ========================================
// QLUAACTIONHELPERS


namespace QLuaActionHelpers {
  
  struct QTIDE_API Connection { 
    QObject *o; 
    const char *s; 
    Connection(QObject *o, const char *s) : o(o),s(s) {} 
  };

  struct QTIDE_API NewAction {
    NewAction(QString t) : text(t), checkable(false), checked(false) {}
    NewAction(QString t, bool b) : text(t), checkable(true), checked(b) {}
    QString text;
    bool checkable;
    bool checked;
  };

  QTIDE_API QAction* operator<<(QAction *a, NewAction b);
  QTIDE_API QAction* operator<<(QAction *action, Connection c);
  QTIDE_API QAction* operator<<(QAction *action, QIcon icon);
  QTIDE_API QAction* operator<<(QAction *action, QActionGroup &group);
  QTIDE_API QAction* operator<<(QAction *action, QKeySequence key);
  QTIDE_API QAction* operator<<(QAction *action, QString s);
  QTIDE_API QAction* operator<<(QAction *action, QVariant v);
  QTIDE_API QAction* operator<<(QAction *action, QAction::MenuRole);
  
}




#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */

