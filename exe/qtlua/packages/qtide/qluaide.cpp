/* -*- C++ -*- */

#include <QtGlobal>
#include <QApplication>
#include <QDebug>
#include <QDesktopServices>
#include <QDir>
#include <QDockWidget>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QList>
#include <QMainWindow>
#include <QMap>
#include <QMenu>
#include <QMenuBar>
#include <QMetaEnum>
#include <QMetaObject>
#include <QMetaMethod>
#include <QPageSetupDialog>
#include <QPainter>
#include <QPointer>
#include <QRegExp>
#include <QSet>
#include <QSettings>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QVariant>
#include <QWhatsThis>

#include "qluaide.h"
#include "qluatextedit.h"
#include "qluaeditor.h"
#include "qluasdimain.h"
#include "qluamdimain.h"
#include "qluabrowser.h"
#include "qluaapplication.h"




// ========================================
// ACTION HELPERS


namespace QLuaActionHelpers {

  QAction * 
  operator<<(QAction *action, QIcon icon)
  {
    action->setIcon(icon);
    return action;
  }
  
  QAction * 
  operator<<(QAction *action, QActionGroup &group)
  {
    action->setActionGroup(&group);
    return action;
  }
  
  QAction * 
  operator<<(QAction *action, QKeySequence shortcut)
  {
    QList<QKeySequence> shortcuts = action->shortcuts();
    shortcuts.prepend(shortcut);
    action->setShortcuts(shortcuts);
    return action;
  }
  
  QAction * 
  operator<<(QAction *action, QString string)
  {
    if (action->text().isEmpty())
      action->setText(string);
    else if (action->statusTip().isEmpty())
      action->setStatusTip(string);
    else if (action->whatsThis().isEmpty())
      action->setWhatsThis(string);
    return action;
  }
  
  QAction *
  operator<<(QAction *action, Connection c)
  {
    QObject::connect(action, SIGNAL(triggered(bool)), c.o, c.s);
    return action;
  }
  
  QAction *
  operator<<(QAction *action, QVariant variant)
  {
    action->setData(variant);
    return action;
  }
  
  QAction *
  operator<<(QAction *action, QAction::MenuRole role)
  {
    action->setMenuRole(role);
    return action;
  }

  QAction* 
  operator<<(QAction *a, NewAction b)
  {
    a->setText(b.text);
    a->setCheckable(b.checkable);
    a->setChecked(b.checked);
    a->setIcon(QIcon());
    a->setToolTip(QString());
    a->setStatusTip(QString());
    a->setShortcuts(QList<QKeySequence>());
    return a;
  }

}


using namespace QLuaActionHelpers;



// ========================================
// PRIVATE



class QLuaIde::Private : public QObject
{
  Q_OBJECT
public:
  QLuaIde *q;
  int      uid;
  bool     editOnError;
  bool     windowsChangedScheduled;
  QObjectList windows;
  QStringList recentFiles;
  QPointer<QtLuaEngine> engine;
  QPointer<QLuaSdiMain> sdiMain;
  QPointer<QLuaMdiMain> mdiMain;
  QPointer<QMenuBar> globalMenuBar;
  QPointer<QWidget> returnTo;
  QString currentPath;
  QByteArray executeWhenAccepting;
  QMap<QByteArray,QAction*> actions;
  QSet<QWidget*> closeSet;
  bool closingDown;
                  
public slots:
  void destroyed(QObject*);
  void newEngine();
  void errorMessage(QByteArray);
  void windowShown(QWidget *w);
  void scheduleWindowsChanged();
  void emitWindowsChanged();
  void updateLuaActions();
  void luaAcceptingCommands(bool);
  void updateWindowMenu();
  void fillWindowMenu();
  void fillRecentMenu();
  void doWindowMenuItem();
  void doRecentMenuItem();
public:
  ~Private();
  Private(QLuaIde *q);
  QLuaEditor *findEditor(QString fname);
  bool eventFilter(QObject *o, QEvent *e);
};


QLuaIde::Private::~Private()
{
  delete globalMenuBar;
}


QLuaIde::Private::Private(QLuaIde *q)
  : QObject(q), 
    q(q), 
    uid(1),
    editOnError(false),
    windowsChangedScheduled(false),
    currentPath(QDir::currentPath()),
    closingDown(false)
{
  QLuaApplication *app = QLuaApplication::instance();
  connect(app, SIGNAL(newEngine(QtLuaEngine*)),
          this, SLOT(newEngine()));
  connect(app, SIGNAL(acceptingCommands(bool)),
          this, SLOT(luaAcceptingCommands(bool)) );
  newEngine();
}


QLuaEditor *
QLuaIde::Private::findEditor(QString fname)
{
  QLuaEditor *e;
  QString cname = QFileInfo(fname).canonicalFilePath();
  if (! cname.isEmpty())
    foreach(QObject *o, windows)
      if ((e = qobject_cast<QLuaEditor*>(o)))
        if (e->fileName() == cname)
          if (! e->isWindowModified())
            return e;
  if (! cname.isEmpty())
    foreach(QObject *o, windows)
      if ((e = qobject_cast<QLuaEditor*>(o)))
        if (e->fileName() == cname)
          return e;
  return 0;
}


void 
QLuaIde::Private::scheduleWindowsChanged()
{
  if (!windowsChangedScheduled)
    QTimer::singleShot(0, this, SLOT(emitWindowsChanged()));
  windowsChangedScheduled = true;
}


void 
QLuaIde::Private::emitWindowsChanged()
{
  windowsChangedScheduled = false;
  emit q->windowsChanged();
}


static QString
incrementName(QString s)
{
  int l = s.size();
  while (l>0 && s[l-1].isDigit())
    l = l - 1;
  int n = s.mid(l).toInt();
  return s.left(l) + QString::number(n+1);
}


void
QLuaIde::Private::windowShown(QWidget *w)
{
  if (! windows.contains(w))
    {
      // find unique name
      QString name = w->objectName();
      if (name.isEmpty())
        name = "window1";
      QStringList list = q->windowNames();
      while (list.contains(name))
        name = incrementName(name);
      w->setObjectName(name);
      // append window
      connect(w, SIGNAL(destroyed(QObject*)),
              this, SLOT(destroyed(QObject*)) );
      windows.append(w);
      // placement policy
      if (!mdiMain || !mdiMain->adopt(w))
        q->loadWindowGeometry(w);
      // name
      QtLuaEngine *engine = QLuaApplication::engine();
      if (engine)
        engine->nameObject(w);
      // advertise changed to windows list
      scheduleWindowsChanged();
    }
}


void
QLuaIde::Private::destroyed(QObject *o)
{
  windows.removeAll(o);
  scheduleWindowsChanged();
}


void
QLuaIde::Private::newEngine()
{
  if (engine)
    disconnect(engine, 0, this, 0);
  engine = QLuaApplication::engine();
  if (! engine)
    return;
  connect(engine, SIGNAL(errorMessage(QByteArray)),
          this, SLOT(errorMessage(QByteArray)) );
  connect(engine, SIGNAL(stateChanged(int)),
          this, SLOT(updateLuaActions()) );
}


void 
QLuaIde::Private::updateLuaActions()
{
  QLuaApplication *app = QLuaApplication::instance();
  bool accepting = app->isAcceptingCommands();
  // stop and pause
  QAction *stopAction = q->hasAction("ActionLuaStop");
  QAction *pauseAction = q->hasAction("ActionLuaPause");
  if (engine && !engine->isReady())
    accepting = false;
  if (pauseAction)
    pauseAction->setEnabled(engine && !accepting);
  if (pauseAction)
    pauseAction->setChecked(engine && engine->isPaused());
  if (stopAction)
    stopAction->setEnabled(!accepting);
  // actions that need lua
  QAction *prefAction = q->hasAction("ActionPreferences");
  QAction *helpAction = q->hasAction("ActionHelp");

  if (prefAction)
    prefAction->setEnabled(engine && engine->runSignalHandlers());
  if (helpAction)
    helpAction->setEnabled(engine && engine->runSignalHandlers());
}


void 
QLuaIde::Private::luaAcceptingCommands(bool accepting)
{
  if (accepting && !executeWhenAccepting.isEmpty())
    {
      QByteArray cmd = executeWhenAccepting;
      executeWhenAccepting = QByteArray();
      q->luaExecute(cmd);
      return;
    }
  updateLuaActions();
}


static bool 
errorMessageFname(QString fname, int lineno, QString msg, int level)
{
  QFileInfo info(fname);
  if (info.exists())
    {
      QLuaEditor *e = QLuaIde::instance()->editor(fname);
      e->widget()->showLine(lineno);
      if (level)
        msg = QLuaIde::Private::tr("Error: called from here.");
      else 
        msg = QLuaIde::Private::tr("Error: ") + msg;
      e->showStatusMessage(msg);
      return true;
    }
  return false;
}  


static bool 
errorMessageEname(QString ename, int lineno, QString msg, int level)
{
  QtLuaEngine *engine = QLuaApplication::engine();
  QObject *o = engine->namedObject(ename);
  QLuaEditor *e = qobject_cast<QLuaEditor*>(o);
  if (e)
    {
      QLuaIde::instance()->activateWidget(e);
      e->widget()->showLine(lineno);
      if (level) 
        msg = QLuaIde::Private::tr("Error: called from here.");
      else 
        msg = QLuaIde::Private::tr("Error: ") + msg;
      e->showStatusMessage(msg);
      return true;
    }
  return false;
}


bool
QLuaIde::Private::eventFilter(QObject *o, QEvent *e)
{
  if (o == sdiMain && e->type() == QEvent::ActivationChange)
    updateWindowMenu();
  return false;
}


void
QLuaIde::Private::errorMessage(QByteArray m)
{
  QtLuaEngine *engine = QLuaApplication::engine();
  if (editOnError && engine)
    {
      QString message = QString::fromLocal8Bit(m.constData());
      if (message.contains('\n'))
        message.truncate(message.indexOf('\n'));
      QStringList location = engine->lastErrorLocation();
      for (int i = location.size()-1; i>=0; --i)
        {
          QString loc = location.at(i);
          QRegExp re3("^@(.+):([0-9]+)$");
          re3.setMinimal(true);
          if (re3.indexIn(loc) >= 0)
            if (errorMessageFname(re3.cap(1), re3.cap(2).toInt(), message, i))
              continue;
          QRegExp re4("^qt\\.(.+):([0-9]+)$");
          re4.setMinimal(true);
          if (re4.indexIn(loc) >= 0)
            if (errorMessageEname(re4.cap(1), re4.cap(2).toInt(), message, i))
              continue;
        }
    }
}


void 
QLuaIde::Private::updateWindowMenu()
{
  QWidget *active = q->activeWindow();
  QAction *prevaction = 0;
  QAction *menuaction = q->hasAction("MenuWindows");
  QMenu *menu = (menuaction) ? menuaction->menu() : 0;
  if (menu)
    {
      foreach(QAction *action, menu->actions())
        if (action->isCheckable())
          {
            QObject *object = qVariantValue<QObject*>(action->data());
            if (object && windows.contains(object))
              {
                QWidget *window = qobject_cast<QWidget*>(object);
                action->setChecked(window && window == active);
                if (window && window == returnTo)
                  prevaction = action;
              }
          }
    }
  QAction *actionReturnToPrevious = q->hasAction("ActionReturnToPrevious");
  if (actionReturnToPrevious)
    {
      QString text = tr("&Return to Previous");
      if (prevaction)
        text = tr("&Return to \"%1\"").arg(prevaction->text());
      actionReturnToPrevious->setText(text);
      actionReturnToPrevious->setVisible(sdiMain && active == sdiMain);
      actionReturnToPrevious->setEnabled(prevaction);
    }
}


void 
QLuaIde::Private::fillWindowMenu()
{
  QAction *menuaction = q->hasAction("MenuWindows");
  QMenu *menu = (menuaction) ? menuaction->menu() : 0;
  if (menu)
    {
      menu->clear();
      menu->addAction(q->stdAction("ActionReturnToPrevious"));
      if (sdiMain)
        menu->addSeparator();
      if (mdiMain)
        {
          menu->addAction(mdiMain->dockAction());
          if (! mdiMain->isTabMode())
            {
              menu->addAction(mdiMain->cascadeAction());
              menu->addAction(mdiMain->tileAction());
            }
          menu->addSeparator();
        }
      // window list
      int k = 0;
      foreach(QObject *o, windows)
        {
          QWidget *w = qobject_cast<QWidget*>(o);
          if (w == 0 || w == mdiMain)
            continue;
          QAction *action = 0;
          QString s = w->windowTitle();
          if (s.isEmpty())
            continue;
          action = menu->addAction(s.replace("[*]",""))
            << qVariantFromValue<QObject*>(o)
            << Connection(this, SLOT(doWindowMenuItem()))
            << tr("Activate the specified window.");
          action->setCheckable(true);
          if (++k < 10)
            action->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_0 + k));
        }
    }
}


void 
QLuaIde::Private::doWindowMenuItem()
{
  QAction *a = qobject_cast<QAction*>(sender());
  if (a)
    {
      QObject *o = qVariantValue<QObject*>(a->data());
      if (o && windows.contains(o))
        {
          QWidget *w = qobject_cast<QWidget*>(o);
          if (w)
            q->activateWidget(w);
        }
    }
}


void 
QLuaIde::Private::fillRecentMenu()
{
  QAction *menuaction = q->hasAction("MenuOpenRecent");
  QMenu *menu = (menuaction) ? menuaction->menu() : 0;
  if (menu)
    {
      menu->clear();
      foreach (QString fname, recentFiles)
        {
          QFileInfo fi(fname);
          QString n = fi.fileName();
          QAction *action = menu->addAction(n);
          QFontMetrics fm(action->font());
          QString p = fm.elidedText(fi.filePath(), Qt::ElideLeft, 300);
          action->setText(QString("%1 [%2]").arg(fi.fileName()).arg(p));
          action->setData(fname);
          action->setStatusTip(tr("Open the named file."));
          connect(action,SIGNAL(triggered()),this,SLOT(doRecentMenuItem()));
        }
      menu->addSeparator();
      QAction *action = menu->addAction(tr("&Clear"));
      action->setStatusTip(tr("Clear the history of recent files."));
      connect(action, SIGNAL(triggered()), q, SLOT(clearRecentFiles()));
    }
}


void 
QLuaIde::Private::doRecentMenuItem()
{
  QAction *action = qobject_cast<QAction*>(sender());
  if (action)
    {
      QString fname = action->data().toString();
      if (! fname.isEmpty())
        q->openFile(fname, false, q->activeWindow());
    }
}




// ========================================
// QLUAIDE



QPointer<QLuaIde> qLuaIde;


QLuaIde *
QLuaIde::instance()
{
  if (! qLuaIde)
    qLuaIde = new QLuaIde();
  return qLuaIde;
}


QLuaIde::QLuaIde()
  : QObject(QCoreApplication::instance()),
    d(new Private(this))
{
  loadRecentFiles();
  setObjectName("qLuaIde");
#ifdef QT_MAC_USE_NATIVE_MENUBAR
  defaultMenuBar(globalMenuBar());
#endif
  connect(QLuaApplication::instance(), SIGNAL(windowShown(QWidget*)),
          d, SLOT(windowShown(QWidget*)) );
  QtLuaEngine *engine = QLuaApplication::engine();
  if (engine)
    engine->nameObject(this);
  // pickup existing visible windows.
  foreach(QWidget *w, QApplication::topLevelWidgets())
    if (w->windowType() == Qt::Window && w->isVisible() 
        && ! w->testAttribute(Qt::WA_DontShowOnScreen) )
      d->windowShown(w);
}


bool 
QLuaIde::editOnError() const
{
  return d->editOnError;
}


bool 
QLuaIde::mdiDefault() const
{
#ifdef Q_WS_WIN
  return true;
#else
  return false;
#endif
}


QLuaSdiMain *
QLuaIde::sdiMain() const
{
  return d->sdiMain;
}


QLuaMdiMain *
QLuaIde::mdiMain() const
{
  return d->mdiMain;
}


QMenuBar *
QLuaIde::globalMenuBar() const
{
  if (! d->globalMenuBar)
    d->globalMenuBar = new QMenuBar(0);
  return d->globalMenuBar;
}


QMenuBar *
QLuaIde::defaultMenuBar(QMenuBar *menu)
{
  menu->clear();
  menu->addAction(stdAction("MenuFile"));
  menu->addAction(stdAction("MenuWindows"));
  menu->addAction(stdAction("MenuHelp"));
  return menu;
}


QObjectList 
QLuaIde::windows() const
{
  return d->windows;
}


QStringList 
QLuaIde::windowNames() const
{
  QStringList s;
  foreach(QObject *o, d->windows)
    if (! o->objectName().isEmpty())
      s += o->objectName();
  return s;
}


QStringList 
QLuaIde::recentFiles() const
{
  return d->recentFiles;
}


void 
QLuaIde::setEditOnError(bool b)
{
  d->editOnError = b;
}

void 
QLuaIde::addRecentFile(QString fname)
{
  d->recentFiles.removeAll(fname);
  d->recentFiles.prepend(fname);
  while(d->recentFiles.size() > 8)
    d->recentFiles.removeLast();
}


void 
QLuaIde::clearRecentFiles()
{
  QSettings s;
  s.remove("editor/recentFiles");
  d->recentFiles.clear();
}


void 
QLuaIde::loadRecentFiles()
{
  QSettings s;
  d->recentFiles = s.value("editor/recentFiles").toStringList();
}


void 
QLuaIde::saveRecentFiles()
{
  QSettings s;
  s.setValue("editor/recentFiles", d->recentFiles);
}


void 
QLuaIde::activateWidget(QWidget *w)
{
  if (w)
    {
      if (! d->mdiMain || ! d->mdiMain->activate(w))
        {
          w = w->window();
          w->show();
          w->raise();
          w->activateWindow();
        }
    }
}


void 
QLuaIde::activateConsole(QWidget *returnTo)
{
  if (returnTo && returnTo != d->sdiMain)
    {
      d->returnTo = returnTo;
      d->scheduleWindowsChanged();
    }
  activateWidget(d->sdiMain);
}


void 
QLuaIde::loadWindowGeometry(QWidget *w)
{
  QString name = w->objectName();
  if (d->windows.contains(w) && !name.isEmpty())
    {
      // sdi or mdi?
      QDockWidget *dw = 0;
      QMdiSubWindow *sw = 0;
      QMainWindow *mw = qobject_cast<QMainWindow*>(w);
      while (w && w->windowType() != Qt::Window && w->parentWidget() 
             && ! (sw = qobject_cast<QMdiSubWindow*>(w)) 
             && ! (dw = qobject_cast<QDockWidget*>(w)) )
        w = w->parentWidget();
      // proceed
      if (mw && !name.isEmpty())
        {
          QSettings s;
          s.beginGroup(sw ? "mdi" : "sdi");
          s.beginGroup(name);
          if (! dw)
            w->restoreGeometry(s.value("geometry").toByteArray());
          mw->restoreState(s.value("state").toByteArray());
        }
    }
}


void 
QLuaIde::saveWindowGeometry(QWidget *w)
{
  QString name = w->objectName();
  if (d->windows.contains(w) && !name.isEmpty())
    {
      // sdi or mdi?
      QDockWidget *dw = 0;
      QMdiSubWindow *sw = 0;
      QMainWindow *mw = qobject_cast<QMainWindow*>(w);
      while (w && w->windowType() != Qt::Window && w->parentWidget() 
             && ! (sw = qobject_cast<QMdiSubWindow*>(w)) 
             && ! (dw = qobject_cast<QDockWidget*>(w)) )
        w = w->parentWidget();
      // proceed
      if (mw && !name.isEmpty())
        {
          QSettings s;
          s.beginGroup(sw ? "mdi" : "sdi");
          s.beginGroup(name);
          s.setValue("state", mw->saveState());
          if (! dw)
            s.setValue("geometry", w->saveGeometry());
        }
    }
}


QWidget *
QLuaIde::previousWindow() const
{
  return d->returnTo;
}


QWidget* 
QLuaIde::activeWindow() const
{
  QWidget *window  = QApplication::activeWindow();
  if (window && window == d->mdiMain)
    window = d->mdiMain->activeWindow();
  if (window && d->windows.contains(window))
    return window;
  return 0;
}


QLuaEditor *
QLuaIde::editor(QString fname)
{
  // find existing
  QLuaEditor *e = d->findEditor(fname);
  if (e)
    {
      activateWidget(e);
    }
  else
    {
      // create
      e = new QLuaEditor();
      e->setAttribute(Qt::WA_DeleteOnClose);
      // load
      if (! fname.isEmpty())
        if (! e->readFile(fname))
          {
            delete e;
            return 0;
          }
      // show
      e->show();
    }
  return e;
}


QLuaInspector *
QLuaIde::inspector()
{
  // TODO
  return 0;
}


QLuaBrowser *
QLuaIde::browser(QString s)
{
  if (QFileInfo(s).exists())
    return browser(QUrl::fromLocalFile(s));
  return browser(QUrl(s));
}


QLuaBrowser *
QLuaIde::browser(QUrl url)
{
  QLuaBrowser *e = new QLuaBrowser();
  e->setAttribute(Qt::WA_DeleteOnClose);
  if (! url.isEmpty())
    e->setUrl(url);
#if HAVE_QTWEBKIT
  e->show();
  activateWidget(e);
#endif
  return e;
}


QLuaSdiMain *
QLuaIde::createSdiMain()
{
  QLuaSdiMain *e = d->sdiMain;
  QtLuaEngine *engine = QLuaApplication::engine();
  if (e)
    {
      activateWidget(e);
    }
  else
    {
      // create
      e = new QLuaSdiMain();
      e->setAttribute(Qt::WA_DeleteOnClose);
      e->installEventFilter(d);
      if (engine)
        engine->nameObject(e);
      // show
      d->sdiMain = e;
      e->show();
    }
  return e;
}


QLuaMdiMain *
QLuaIde::createMdiMain()
{
  QLuaMdiMain *m = d->mdiMain;
  QtLuaEngine *engine = QLuaApplication::engine();
  if (m)
    {
      activateWidget(m);
    }
  else 
    {
      // create
      m = new QLuaMdiMain();
      if (engine)
        engine->nameObject(m);
      // we do not want that:
      // if it contains the console, we'll exit anyway.
      // if it does not contain the console, close hides and adopt shows.
      m->setAttribute(Qt::WA_DeleteOnClose,false);
      // show
      d->mdiMain = m;
    }
  return m;
}



QAction *
QLuaIde::newAction(QString title)
{
  QAction *action = new QAction(title, this);
  action->setMenuRole(QAction::NoRole);
  return action;
}


QMenu *
QLuaIde::newMenu(QString title)
{
  QMenu *menu = new QMenu();
  QAction *action = new QAction(title, this);
  action->setMenu(menu);
  action->setMenuRole(QAction::NoRole);
  connect(action, SIGNAL(destroyed()), menu, SLOT(deleteLater()));
  return menu;
}


QAction *
QLuaIde::hasAction(QByteArray name)
{
  if (d->actions.contains(name))
    return d->actions[name];
  return 0;
}
 
 
QAction *
QLuaIde::stdAction(QByteArray name)
{
  // already there
  QAction *action = 0;
  if (d->actions.contains(name))
    return d->actions[name];
  // menus
  if (name == "MenuFile")
    {
      QMenu *menu = newMenu(tr("&File","file|"));
      menu->addAction(stdAction("ActionFileNew"));
      menu->addAction(stdAction("ActionFileOpen"));
      menu->addAction(stdAction("MenuOpenRecent"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionPreferences"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionFileClose"));
      menu->addAction(stdAction("ActionFileQuit"));
      action = menu->menuAction();
     }
  else if (name == "MenuWindows")
    {
      QMenu *menu = newMenu(tr("&Windows","windows|"));
      connect(menu, SIGNAL(aboutToShow()),
              d, SLOT(updateWindowMenu()) );
      connect(this, SIGNAL(windowsChanged()), 
              d, SLOT(fillWindowMenu()) );
      action = menu->menuAction();
      d->fillWindowMenu();
    }
  else if (name == "MenuHelp")
    {
      QMenu *menu = newMenu(tr("&Help", "help|"));
      menu->addAction(stdAction("ActionHelp"));
      menu->addSeparator();     
      menu->addAction(stdAction("ActionAbout"));
      menu->addAction(stdAction("ActionAboutQt"));
      action = menu->menuAction();
    }
  else if (name == "MenuOpenRecent")
    {
      QMenu *menu = newMenu(tr("Open &Recent","file|recent"));
      connect(menu, SIGNAL(aboutToShow()), d, SLOT(fillRecentMenu()));
      action = menu->menuAction()
        << QIcon(":/images/filerecent.png")
        << tr("Open a file into a new window.");
    }
  // actions
  else if (name == "ActionFileNew")
    {
      action = newAction(tr("&New","file|new"))
        << QKeySequence(QKeySequence::New)
        << QIcon(":/images/filenew.png")
        << Connection(this, SLOT(doNew()))
        << tr("Create a new text editor window.");
    }
  else if (name == "ActionFileOpen")
    {
      action = newAction(tr("&Open", "file|open"))
        << QKeySequence(QKeySequence::Open)
        << QIcon(":/images/fileopen.png")
        << Connection(this, SLOT(doOpen()))
        << tr("Open a file into a new window.");
    }
  else if (name == "ActionFileClose")
    {
      action = newAction(tr("&Close", "file|close"))
        << QKeySequence(QKeySequence::Close)
        << QIcon(":/images/fileclose.png")
        << Connection(this, SLOT(doClose()))
        << tr("Close this window.");
    }
  else if (name == "ActionFileQuit")
    {
      action = newAction(tr("&Quit", "file|quit"))
        << QIcon(":/images/filequit.png")
        << Connection(this, SLOT(doQuit()))
        << tr("Quit the application.")
#ifndef Q_WS_MAC
        << QKeySequence(tr("Ctrl+Q", "file|quit"))
#endif
        << QAction::QuitRole;
    }
  else if (name == "ActionReturnToPrevious")
    {
      action = newAction(tr("&Return to Previous","windows|returnto"))
        << QKeySequence(tr("F4","windows|returntoprevious"))
        << QKeySequence(tr("F5","windows|returntoprevious"))
        << Connection(this, SLOT(doReturnToPrevious()))
        << tr("Return to previously active window.");
      action->setVisible(false);
    }
  else if (name == "ActionPreferences")
    {
      action = newAction(tr("&Preferences", "edit|prefs")) 
        << QIcon(":/images/editprefs.png")
        << Connection(this, SLOT(doPreferences()))
        << tr("Show the preference dialog.")
        << QAction::PreferencesRole;
    }
  else if (name == "ActionLuaStop")
    {
      action = newAction(tr("&Stop","lua|stop"))
        << QIcon(":/images/stop.png")
#ifdef Q_WS_MAC
        << QKeySequence(tr("Ctrl+Pause","lua|stop"))
        << QKeySequence(tr("Ctrl+.","lua|stop"))
#else
        << QKeySequence(tr("Ctrl+.","lua|stop"))
        << QKeySequence(tr("Ctrl+Pause","lua|stop"))
#endif
        << Connection(this, SLOT(doLuaStop()))
        << tr("Stop the execution of the current Lua command.");
    }
  else if (name == "ActionLuaPause")
    {
      action = newAction(tr("&Pause","lua|pause"))
        << QIcon(":/images/playerpause.png")
        << QKeySequence(tr("ScrollLock","lua|pause"))
        << Connection(this, SLOT(doLuaPause()))
        << tr("Suspend or resume the execution of the current Lua command.");
      action->setCheckable(true);
    }
  else if (name == "ActionHelp")
    {
      action = newAction(tr("Help Index", "help|help"))
        << QKeySequence(tr("F1", "help|help"))
        << QIcon(":/images/helpindex.png")
        << Connection(this, SLOT(doHelp()))
        << tr("Opens the help window.");
    }
  else if (name == "ActionAbout")
    {
      QString name = QCoreApplication::applicationName();
      action = newAction(tr("About %1", "help|aboutqtlua").arg(name))
        << tr("Display information about %1.").arg(name)
        << Connection(QLuaApplication::instance(), SLOT(about()))
        << QAction::AboutRole;
    }
  else if (name == "ActionAboutQt")
    {
      action = newAction(tr("About Qt", "help|aboutqt"))
        << tr("Display information about Qt.")
        << Connection(QLuaApplication::instance(), SLOT(aboutQt()))
        << QAction::AboutQtRole;
    }
  else if (name == "ActionWhatsThis")
    {
      action = QWhatsThis::createAction();
    }
  // cache
  if (action)
    d->actions[name] = action;
  return action;
}


void 
QLuaIde::updateActions()
{
  d->updateLuaActions();
}


bool
QLuaIde::openFile(QString fileName, bool inOther, QWidget *window)
{
  bool okay = false;
  const QMetaObject *mo = (window) ? window->metaObject() : 0;
  // try calling a method 'bool openFile(QString, bool)'
  if (mo && mo->invokeMethod(window, "openFile", 
                             Q_RETURN_ARG(bool, okay), 
                             Q_ARG(QString, fileName),
                             Q_ARG(bool, inOther) ))
    if (okay)
      return okay;
  // open ourselves
  QString suffix = QFileInfo(fileName).suffix();
  if (suffix == "html" || suffix == "HTML")
    okay = browser(fileName);
  else
    okay = editor(fileName);
  return okay;
}

  
bool 
QLuaIde::newDocument(QWidget *window)
{
  bool okay = false;
  const QMetaObject *mo = (window) ? window->metaObject() : 0;
  // try calling a method 'bool newDocument(QString)'
  if (mo && mo->invokeMethod(window, "newDocument", 
                             Q_RETURN_ARG(bool, okay) ))
    if (okay)
      return okay;
  // open a new text document
  okay = editor();
  return okay;
}



bool
QLuaIde::luaExecute(QByteArray cmd)
{
  QLuaApplication *app = QLuaApplication::instance();
  if (! cmd.simplified().isEmpty())
    return app->runCommand(cmd);
  return false;
}


bool
QLuaIde::luaRestart(QByteArray cmd)
{
  QLuaApplication *app = QLuaApplication::instance();
  if (app->isAcceptingCommands())
    {
      foreach(QObject *object, windows())
        if (object->isWidgetType())
          saveWindowGeometry(static_cast<QWidget*>(object));
      d->executeWhenAccepting = cmd;
      app->restart();
      return true;
    }
  return false;
}


bool 
QLuaIde::quit(QWidget *r)
{
  bool okay = true;
  if (! d->closingDown)
    {
      // confirm
      QString appName = QCoreApplication::applicationName();
      if ( ! QLuaApplication::instance()->isClosingDown())
        if (QMessageBox::question((r) ? r : d->sdiMain, 
                                  tr("Really Quit?"), 
                                  tr("Really quit %0?").arg(appName),
                                  QMessageBox::Ok|QMessageBox::Cancel,
                                  QMessageBox::Cancel) != QMessageBox::Ok )
          return false;
      // mdi settings
      if (d->mdiMain)
        d->mdiMain->saveMdiSettings();
      // close all windows but sdimain and mdimain
      d->closingDown = true;
      d->closeSet.clear();
      d->closeSet += 0;
      d->closeSet += d->sdiMain;
      d->closeSet += d->mdiMain;
      QObjectList wl = d->windows;
      while (okay && wl.size())
        {
          QWidget *w = qobject_cast<QWidget*>(wl.takeFirst());
          if (d->closeSet.contains(w))
            continue;
          d->closeSet += w;
          okay = w->close();
          wl = d->windows;
        }
      if (okay && d->sdiMain)
        okay = d->sdiMain->close();
      if (okay && d->mdiMain)
        okay = d->mdiMain->close();
      if (okay)
        okay = QLuaApplication::instance()->close();
      d->closingDown = okay;
    }
  return okay;
}


int
QLuaIde::messageBox(QString t, QString m, 
                    QMessageBox::StandardButtons buttons,
                    QMessageBox::StandardButton def,
                    QMessageBox::Icon icon)
{
  QWidget *r = activeWindow();
  QMessageBox box(icon, t, m, buttons, r);
  box.setDefaultButton(def);
  return box.exec();
}


QByteArray
QLuaIde::messageBox(QString t, QString m, QByteArray buttons,
                    QByteArray def, QByteArray icon)
{
  const QMetaObject *mo = &QMessageBox::staticMetaObject;
  int imeb = mo->indexOfEnumerator("StandardButtons");
  int imei = mo->indexOfEnumerator("Icon");
  const QMetaEnum meb = mo->enumerator(imeb);
  const QMetaEnum mei = mo->enumerator(imei);
  if (meb.isValid() && mei.isValid())
    {
      QMessageBox::StandardButtons b
        = (QMessageBox::StandardButtons)meb.keysToValue(buttons.constData());
      QMessageBox::StandardButton d
        = (QMessageBox::StandardButton)meb.keyToValue(def.constData());
      QMessageBox::Icon i
        = (QMessageBox::Icon)mei.keyToValue(icon.constData());
      if ((int)b>=0 && (int)d>=0 && (int)i>=0)
        return meb.valueToKey(messageBox(t,m,b,d,i));
    }
  return QByteArray("<invalidargs>");
}


QString 
QLuaIde::fileDialogFilters()
{
  bool needHtml = true;
  QStringList filters;
  foreach(QLuaTextEditModeFactory *mode, 
          QLuaTextEditModeFactory::factories())
    {
      filters += mode->filter();
      if (mode->suffixes().contains("html"))
        needHtml = false;
    }
  if (needHtml)
    filters += htmlFilesFilter();
  filters += allFilesFilter();
  return filters.join(";;");
}


QString 
QLuaIde::htmlFilesFilter()
{
  return tr("HTML Files (*.html)");
}


QString 
QLuaIde::allFilesFilter()
{
  return tr("All Files (*)");
}


void 
QLuaIde::doNew()
{
  newDocument(activeWindow());
}


void 
QLuaIde::doOpen()
{
  QWidget *w = activeWindow();
  QString m = tr("Open File");
  QString p = d->currentPath;
  QString f = fileDialogFilters();
  QString s = allFilesFilter();
  // get directory
  QString z = (w) ? w->property("fileName").toString() : QString();
  if (! z.isEmpty())
    p = QFileInfo(z).absolutePath();
  // proceed
  QFileDialog::Options o = QFileDialog::DontUseNativeDialog;
  QStringList files = QFileDialog::getOpenFileNames(w, m, p, f, &s, o);
  bool inOther = false;
  foreach(QString fname, files)
    if (! fname.isEmpty())
      {
        d->currentPath = QFileInfo(fname).absolutePath();
        openFile(fname, inOther, w);
        inOther = true;
      }
}


bool 
QLuaIde::doClose()
{
  QWidget *window = activeWindow();
  if (window)
    return window->close();
  return false;
}


bool 
QLuaIde::doQuit()
{
  return quit(activeWindow());
}


void 
QLuaIde::doReturnToConsole()
{
  activateConsole(activeWindow());
}


void 
QLuaIde::doReturnToPrevious()
{
  QWidget *w = activeWindow();
  if (w && w == d->sdiMain && d->returnTo)
    activateWidget(d->returnTo);
  else if (w != d->sdiMain)
    activateConsole(w);
}


void 
QLuaIde::doLuaStop()
{
  if (d->engine)
    d->engine->stop(true);
}


void 
QLuaIde::doLuaPause()
{
  QAction *action = stdAction("ActionLuaPause");
  if (action && d->engine)
    {
      if (action->isChecked())
        d->engine->stop(false);
      else
        d->engine->resume(false);
    }
}


void 
QLuaIde::doPreferences()
{
  emit prefsRequested(activeWindow());
}


void 
QLuaIde::doHelp()
{
  emit helpRequested(activeWindow());
}


// ========================================
// MOC


#include "qluaide.moc"





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
