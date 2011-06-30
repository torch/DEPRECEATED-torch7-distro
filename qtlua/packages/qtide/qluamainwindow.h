// -*- C++ -*-

#ifndef QLUAMAINWINDOW_H
#define QLUAMAINWINDOW_H

#include "qtide.h"


#include <QAction>
#include <QActionGroup>
#include <QFile>
#include <QIcon>
#include <QKeySequence>
#include <QMainWindow>
#include <QMessageBox>
#include <QObject>
#include <QWidget>


class QLuaEditor;
class QLuaTextEdit;
class QLuaTextEditModeFactory;
class QMenu;
class QMenuBar;
class QStatusBar;
class QToolBar;
class QUrl;




// Text Editor

class QTIDE_API QLuaMainWindow : public QMainWindow
{
  Q_OBJECT
public:
  ~QLuaMainWindow();
  QLuaMainWindow(QString objName, QWidget *parent=0);
  Q_INVOKABLE QAction *hasAction(QByteArray what);
  Q_INVOKABLE QAction *stdAction(QByteArray what);
  virtual QAction *createAction(QByteArray name);
  virtual QMenuBar *menuBar();
  virtual QMenuBar  *createMenuBar();
  virtual QToolBar *toolBar();
  virtual QToolBar *createToolBar();
  virtual QStatusBar *statusBar();
  virtual QStatusBar *createStatusBar();
  virtual QPrinter *loadPageSetup();
  virtual void savePageSetup();
  virtual void loadSettings();
  virtual void saveSettings();
  virtual bool canClose();
                              
public slots:
  virtual void updateActions();
  virtual void updateActionsLater();
  virtual void clearStatusMessage();
  virtual void showStatusMessage(const QString & message, int timeout=0);
  virtual bool openFile(QString fileName, bool inOther=false);
  virtual bool newDocument();
 protected:
  virtual void closeEvent(QCloseEvent *e);
  QAction *newAction(QString title);
  QAction *newAction(QString title, bool checked);
  QMenu   *newMenu(QString title);
public:
  class Private;
private:
  Private *d;
};





#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */

