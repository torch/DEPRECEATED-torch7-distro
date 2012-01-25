// -*- C++ -*-

#ifndef QLUAMDIMAIN_H
#define QLUAMDIMAIN_H

#include "qtide.h"
#include "qluamainwindow.h"

#include <QAction>
#include <QFile>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QObject>
#include <QString>
#include <QTextCharFormat>
#include <QWidget>



// MDI Wrapper

class QTIDE_API QLuaMdiMain : public QLuaMainWindow
{
  Q_OBJECT
  Q_PROPERTY(bool tabMode READ isTabMode WRITE setTabMode)
  Q_PROPERTY(QByteArray clientClass READ clientClass WRITE setClientClass)

public:
  ~QLuaMdiMain();
  QLuaMdiMain(QWidget *parent=0);

  Q_INVOKABLE QMdiArea* mdiArea();
  Q_INVOKABLE bool isActive(QWidget *w);
  Q_INVOKABLE bool isTabMode() const;
  Q_INVOKABLE QByteArray clientClass() const;
  Q_INVOKABLE QWidget *activeWindow() const;

  virtual bool canClose();
  virtual void loadSettings();
  virtual void saveMdiSettings();
  virtual QMenuBar *menuBar();
  virtual QToolBar *createToolBar();
  virtual QMenuBar  *createMenuBar();
  virtual QStatusBar *createStatusBar();
  QAction *tileAction();
  QAction *cascadeAction();
  QAction *tabModeAction();
  QAction *dockAction();

public slots:
  bool activate(QWidget *w);
  bool adopt(QWidget *w);
  void adoptAll();
  void setTabMode(bool b);
  void setClientClass(QByteArray c);
  void doNew();
  
public:
  class Client;
  class Private;
  class Shell;
private:
  Private *d;
};


#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */

