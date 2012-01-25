/* -*- C++ -*- */

#include <QtGlobal>
#include <QAction>
#include <QApplication>
#include <QDebug>
#include <QDockWidget>
#include <QLayout>
#include <QLayoutItem>
#include <QMainWindow>
#include <QMap>
#include <QMessageBox>
#include <QMenuBar>
#include <QMoveEvent>
#include <QPointer>
#include <QRegExp>
#include <QResizeEvent>
#include <QSet>
#include <QSettings>
#include <QStatusBar>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QVariant>
#include <QWindowStateChangeEvent>

#include "qluaide.h"
#include "qluamainwindow.h"
#include "qluaeditor.h"
#include "qluasdimain.h"
#include "qluamdimain.h"

#include "qluaapplication.h"


#ifdef QT_MAC_STEAL_STATUSBAR
# define QT_STEAL_STATUSBAR QT_MAC_STEAL_STATUSBAR
#else
# define QT_STEAL_STATUSBAR 1
#endif



// ========================================
// CLASSES


class QLuaMdiMain::Private : public QObject 
{
  Q_OBJECT
public:
  Private(QLuaMdiMain *parent);
  bool eventFilter(QObject *watched, QEvent *event);
public slots:
  void dockCurrentWindow(bool b);
  void setActive(Client *client);
  void focusChange(QWidget *old, QWidget *now);
  void guiUpdateLater();
  void guiUpdate();
public:
  QLuaMdiMain *q;
  QMdiArea *area;
  QList<Client*> clients;
  QMap<QWidget*,Client*> table;
  QPointer<Client> active;
  bool closingDown;
  bool guiUpdateScheduled;
  bool tabMode;
  QByteArray clientClass;
  QPointer<QMenu> fileMenu;
  QPointer<QMenu> editMenu;
  QPointer<QAction> tileAction;
  QPointer<QAction> cascadeAction;
  QPointer<QAction> dockAction;
};


class QLuaMdiMain::Client : public QObject 
{
  Q_OBJECT
public:
  ~Client();
  Client(QWidget *widget, Private *parent);
  bool computeWindowTitle();
  virtual bool eventFilter(QObject *watched, QEvent *event);
  void setVisible(bool);
  void deActivate();
  void reActivate();
  bool copyMenuBar(QMenuBar *mb);
public slots:
  void destroyed(QObject*);
  void stealMenuBar();
  void stealStatusBar();
  void disown();
  void fillToolMenu();
  void setupAsSubWindow();
  void setupAsDockWidget();
  void toplevelChanged(bool b);
public:
  Private *d;
  QPointer<QWidget> widget;
  QPointer<QMdiSubWindow> subWindow;
  QPointer<QDockWidget> dockWidget;
  QPointer<QMenuBar> menuBar;
#ifdef QT_STEAL_STATUSBAR
  QPointer<QStatusBar> statusBar;
#endif
};




// ========================================
// SHELL



// We insert this between the qmdisubwindow and a plain qwidget
// in order to reinterpret geometry changes and state changes.
// Also because qmdisubwindow dislikes widgets without layouts
// and without proper size hints. 

class QLuaMdiMain::Shell : public QWidget
{
  Q_OBJECT
  QPointer<QWidget> w;
  bool settingGeometry;
  bool fullScreen;
public:
  Shell(QWidget *w, QWidget *p = 0);
  virtual QSize sizeHint () const;
  virtual bool eventFilter(QObject *o, QEvent *e);
  virtual void resizeEvent(QResizeEvent *e);
};


QLuaMdiMain::Shell::Shell(QWidget *w, QWidget *p)
  : QWidget(p), w(w), settingGeometry(false), fullScreen(false)
{
  w->setParent(this);
  w->setWindowFlags((w->windowFlags() & ~Qt::WindowType_Mask) | Qt::Widget);
  w->setWindowState(Qt::WindowNoState);
  w->installEventFilter(this);
  setWindowModified(w->isWindowModified());
  setWindowTitle(w->windowTitle());
  setFocusProxy(w);
  w->move(0,0);
}


QSize
QLuaMdiMain::Shell::sizeHint() const
{
  QSize size;
  if (w)
    size = w->sizeHint();
  if (w && !size.isValid() && !w->layout())
    size = w->size();
  return size;
}


void 
QLuaMdiMain::Shell::resizeEvent(QResizeEvent *e)
{
  if (w && !fullScreen && w->size() != e->size())
    {
      settingGeometry = true;
      w->setGeometry(rect());
      settingGeometry = false;
    }
}


bool 
QLuaMdiMain::Shell::eventFilter(QObject *o, QEvent *e)
{
  if (o != w)
    return false;

  switch(e->type())
    {
    case QEvent::Resize:
      if (! settingGeometry && ! fullScreen)
        {
          updateGeometry();
          QWidget *parent = parentWidget();
          while (parent && !parent->inherits("QMdiSubWindow"))
            parent = parent->parentWidget();
          if (parent && !(windowState() & ~Qt::WindowActive))
            parent->adjustSize();
        }
      break;
    case QEvent::Move:
      if (! settingGeometry && ! fullScreen)
        {
          QPoint pos = w->pos();
          settingGeometry = true;
          w->move(0,0);
          settingGeometry = false;
          QWidget *parent = parentWidget();
          while (parent && !parent->inherits("QMdiSubWindow"))
            parent = parent->parentWidget();
          if (parent)
            parent->move(pos);
        }
      break;
    case QEvent::WindowStateChange:
      if (! settingGeometry &&
          ! static_cast<QWindowStateChangeEvent*>(e)->isOverride() )
        {
          Qt::WindowStates newState = w->windowState();
          Qt::WindowFlags flags = w->windowFlags();
          settingGeometry = true;
          if ((newState & Qt::WindowFullScreen) && !fullScreen)
            {
              fullScreen = true;
              w->setWindowFlags((flags & ~Qt::WindowType_Mask) | Qt::Window);
              w->showFullScreen();
              newState = Qt::WindowMinimized;
            }
          else if (!(newState & Qt::WindowFullScreen) && fullScreen)
            {
              w->setWindowFlags((flags & ~Qt::WindowType_Mask) | Qt::Widget);
              w->setGeometry(rect());
              fullScreen = false;
            }
          setWindowState(newState);
          settingGeometry = false;
        }
      break;
    default:
      break;
    }
  return false;
}







// ========================================
// PRIVATE IMPLEMENTATION


QLuaMdiMain::Private::Private(QLuaMdiMain *parent)
  : QObject(parent), 
    q(parent),
    closingDown(false),
    guiUpdateScheduled(false),
    tabMode(false)
{
  area = new QMdiArea(q);
  area->installEventFilter(this);
  connect(QApplication::instance(), SIGNAL(focusChanged(QWidget*,QWidget*)),
          this, SLOT(focusChange(QWidget*,QWidget*)) );
}


bool
QLuaMdiMain::Private::eventFilter(QObject *watched, QEvent *event)
{
  if (watched != area)
    return false;
  if (event->type() == QEvent::Resize)
    {
      QRect ar = area->rect().adjusted(+20,+20,-20,-20);
      foreach(Client *client, clients)
        if (client->subWindow && 
            ! ar.intersects(client->subWindow->frameGeometry()) )
          {
            // move window to be at least partially visible
            QRect gr = client->subWindow->geometry();
            gr.moveRight(qMax(gr.right(), ar.left()));
            gr.moveLeft(qMin(gr.left(), ar.right()));
            gr.moveBottom(qMax(gr.bottom(), ar.top()));
            gr.moveTop(qMin(gr.top(), ar.bottom()));
            client->subWindow->setGeometry(gr);
          }
    }
  if (event->type() == QEvent::WindowActivate)
    guiUpdateLater();
  return false;
}


void 
QLuaMdiMain::Private::focusChange(QWidget *old, QWidget *now)
{
  QWidget *w = now;
  while (w && !table.contains(w) && w != q)
    w = w->parentWidget();
  if (w && table.contains(w))
    setActive(table[w]);
}


void 
QLuaMdiMain::Private::setActive(Client *client)
{
  if (active != client)
    {
      guiUpdateLater();
      if (active)
        active->deActivate();
      active = client;
      if (active)
        active->reActivate();
    }
}

void 
QLuaMdiMain::Private::guiUpdateLater()
{
  if (! guiUpdateScheduled && ! closingDown)
    {
      guiUpdateScheduled = true;
      QTimer::singleShot(1, this, SLOT(guiUpdate()));
    }
}


void 
QLuaMdiMain::Private::guiUpdate()
{
  guiUpdateScheduled = false;
  if (closingDown)
    return;
  
  // maximize in tab mode
  if (tabMode && active && active->subWindow && active->widget)
    if (! (active->widget->windowState() & Qt::WindowFullScreen) &&
        ! (active->widget->windowState() & Qt::WindowMaximized) )
      active->subWindow->showMaximized();
  
  // menubar
  QMenuBar *menubar = q->menuBar();
  foreach(QAction *a, menubar->actions())
    menubar->removeAction(a);
  if (! active || ! active->copyMenuBar(menubar))
    QLuaIde::instance()->defaultMenuBar(menubar);
  // dockaction
  QAction *da = q->dockAction();
  da->setEnabled(active != 0);
  da->setChecked(active && active->dockWidget!=0);
  // mdi window title
  bool modified = false;
  QString title = QApplication::applicationName();
  if (active && active->widget)
    {
      modified = active->widget->isWindowModified();
      QString wtitle = active->widget->windowTitle();
      title = wtitle.replace(QRegExp(" -+ .*$"), " - ") + title;
    }
  q->setWindowTitle(title);
  q->setWindowModified(modified);
}


void 
QLuaMdiMain::Private::dockCurrentWindow(bool b)
{
  if (active)
    {
      if (b && !active->dockWidget)
        active->setupAsDockWidget();
      else if (!b && active->dockWidget) 
        active->setupAsSubWindow();
    }
}







// ========================================
// CLIENT IMPLEMENTATION


QLuaMdiMain::Client::~Client()
{
  QWidget *w = widget;
  if (w)
    {
      QLuaIde *ide = QLuaIde::instance();
      bool visible = w->isVisibleTo(w->parentWidget());
      disown();
      if (d->active == this)
        deActivate();
      if (menuBar && !menuBar->actions().isEmpty())
        menuBar->show();
#ifdef QT_STEAL_STATUSBAR
      if (statusBar && qobject_cast<QMainWindow*>(w))
        static_cast<QMainWindow*>(w)->setStatusBar(statusBar);
      if (statusBar)
        statusBar->show();
      statusBar = 0;
#endif
      if (ide)
        ide->loadWindowGeometry(w);
      w->setVisible(visible);
      menuBar = 0;
    }
  d->table.remove(w);
  d->table.remove(subWindow);
  d->table.remove(dockWidget);
  delete subWindow;
  delete dockWidget;
  d->clients.removeAll(this);
  if (d->clients.isEmpty())
    d->q->hide();
#ifdef QT_STEAL_STATUSBAR
  delete statusBar;
#endif
  delete menuBar;
  d->guiUpdateLater();
}


QLuaMdiMain::Client::Client(QWidget *w, Private *parent)
  : QObject(parent),
    d(parent),
    widget(w)
{
  // make sure we can embed this widget
  w->installEventFilter(this);
  connect(w, SIGNAL(destroyed(QObject*)), this, SLOT(destroyed(QObject*)));
  // record our existence
  d->clients.append(this);
  d->table[w] = this;
  // steal menubar and statusbar
  stealMenuBar();
  stealStatusBar();
}


void 
QLuaMdiMain::Client::destroyed(QObject *o)
{
  if (d->active == this)
    d->setActive(0);
  deleteLater();
}


bool
QLuaMdiMain::Client::computeWindowTitle()
{
  if (widget)
    {
      bool modified = widget->isWindowModified();
      QString title = widget->windowTitle();
      QWidget *w = 0;
      if (d->tabMode && subWindow)
        {
          if (title.contains("[*] - "))
            title.replace(QRegExp("\\[\\*\\] - .*$"), "[*]");
        }
      if (subWindow)
        w = subWindow;
      else if (dockWidget)
        w = dockWidget;
      if (w)
        {
#if QT_VERSION >= 0x40500
          w->setWindowTitle(title);
          w->setWindowModified(modified);
#else          
          if (title.contains("[*]"))
            title.replace("[*]", modified ? "*" : "");
          w->setWindowTitle(title);
          if (d && d->q && this == d->active)
            d->q->setWindowModified(modified);
#endif
#if QT_VERSION >= 0x40400
          w->setWindowFilePath(widget->windowFilePath());
#endif
        }
    }
  return false;
}


bool 
QLuaMdiMain::Client::eventFilter(QObject *watched, QEvent *event)
{
  // watched == widget
  if (watched == widget)
    {
      switch(event->type())
        {
        case QEvent::HideToParent:
          setVisible(false);
          break;
        case QEvent::ShowToParent:
          setVisible(true);
          break;
        case QEvent::WindowTitleChange:
        case QEvent::ModifiedChange:
          return computeWindowTitle();
        default:
          break;
        }
    }
  // watched == subWindow
  else if (watched == subWindow)
    {
      if (widget && event->type() == QEvent::WindowStateChange &&
          ! static_cast<QWindowStateChangeEvent*>(event)->isOverride() )
        {
          Qt::WindowStates nstate = subWindow->windowState();
          Qt::WindowStates ostate = widget->windowState();
          if ((nstate & Qt::WindowMinimized) && 
              (ostate & Qt::WindowFullScreen) )
            nstate = (nstate & ~Qt::WindowMinimized) | Qt::WindowFullScreen;
          if (nstate != ostate)
            widget->setWindowState(nstate);
        }
      else if (widget && event->type() == QEvent::Close)
        {
          if (! widget->close())
            event->ignore();
          d->guiUpdate();
          return true;
        }
    }
  // watched == dockWidget
  else if (watched == dockWidget)
    {
      if (widget && event->type() == QEvent::Close)
        {
          if (! widget->close())
            event->ignore();
          d->guiUpdate();
          return true;
        }
    }
  // watched == menuBar
  else if (watched == menuBar)
    {
      switch(event->type())
        {
        case QEvent::ActionAdded:
        case QEvent::ActionChanged:
        case QEvent::ActionRemoved:
          d->guiUpdateLater();
          break;
        default:
          break;
        }
    }
  return false;
}


void
QLuaMdiMain::Client::setVisible(bool show)
{
  if (! widget)
    show = false;
  if (subWindow && show && d->tabMode)
    subWindow->showMaximized();
  else if (subWindow)
    subWindow->setVisible(show);
  else if (dockWidget)
    dockWidget->setVisible(show);
  if (! show && d->active == this)
    d->setActive(0);
}


void 
QLuaMdiMain::Client::disown()
{
  QLuaIde *ide = QLuaIde::instance();
  if (widget && widget->parentWidget())
    {
      bool visible = widget->isVisibleTo(widget->parentWidget());
      if (subWindow)
        ide->saveWindowGeometry(widget);
      widget->hide();
      widget->setParent(0);
      d->table.remove(subWindow);
      d->table.remove(dockWidget);
      if (dockWidget)
        dockWidget->deleteLater();
      if (subWindow)
        subWindow->deleteLater();
    }
}


void 
QLuaMdiMain::Client::fillToolMenu()
{
  QMenu *menu = qobject_cast<QMenu*>(sender());
  QMainWindow *main = qobject_cast<QMainWindow*>(widget);
  if (menu && widget)
    {
      menu->clear();
      QMenu *popup = main->createPopupMenu();
      if (popup)
        foreach (QAction *action, popup->actions())
          menu->addAction(action);
      delete popup;
    }
}


void 
QLuaMdiMain::Client::setupAsSubWindow()
{
  QLuaIde *ide = QLuaIde::instance();
  if (! widget)
    return;
  if (dockWidget)
    disown();
  if (! subWindow)
    {
      QWidget *w = new Shell(widget);
      // create subwindow
      subWindow = d->area->addSubWindow(w);
      d->table[subWindow] = this;
      // mark widget as a simple widget
      Qt::WindowFlags flags = widget->windowFlags() & ~Qt::WindowType_Mask;
      widget->setWindowFlags(flags | Qt::Widget);
      // subwindow callbacks
      subWindow->installEventFilter(this);
      // prepare toolbar action
      QMenu *toolMenu = new QMenu(subWindow);
      connect(toolMenu, SIGNAL(aboutToShow()), this, SLOT(fillToolMenu()));
      QAction *toolAction = new QAction(tr("Toolbars"), subWindow);
      toolAction->setMenu(toolMenu);
      // prepare dock action
      QAction *dockAction = new QAction(tr("Dock"), subWindow);
      dockAction->setCheckable(true);
      connect(dockAction, SIGNAL(triggered(bool)),
              this, SLOT(setupAsDockWidget()) );
      // tweak system menu
      QMenu *menu = subWindow->systemMenu();
      QKeySequence cseq(QKeySequence::Close);
      foreach (QAction *action, menu->actions())
        {
          if (action->shortcut() == cseq)
            action->setShortcut(QKeySequence());
          if ((action->shortcut() == cseq || action->isSeparator() ) ) 
            {
              if (dockAction)
                menu->insertAction(action, dockAction);
              dockAction = 0;
              if (toolAction && qobject_cast<QMainWindow*>(widget))
                menu->insertAction(action, toolAction);
              toolAction = 0;
            }
        }
      // show
      computeWindowTitle();
      ide->loadWindowGeometry(w);
      d->guiUpdateLater();
      widget->show();
      subWindow->show();
      // set focus
      d->setActive(this);
      QWidget *fw = widget;
      fw = fw->focusWidget() ? fw->focusWidget() : fw;
      fw->setFocus(Qt::OtherFocusReason);
    }
}


void 
QLuaMdiMain::Client::setupAsDockWidget()
{
  QLuaIde *ide = QLuaIde::instance();
  if (! widget)
    return;
  if (subWindow)
    disown();
  if (! dockWidget)
    {
      QWidget *w = new Shell(widget);
      // create
      dockWidget = new QDockWidget(widget->windowTitle(), d->q);
      QDockWidget::DockWidgetFeatures f = dockWidget->features();
      dockWidget->setFeatures(f | QDockWidget::DockWidgetVerticalTitleBar);
      dockWidget->setObjectName(widget->objectName());
      dockWidget->setWidget(w);
      d->table[dockWidget] = this;
      // mark widget as a subwindow (to isolate shortcuts)
      Qt::WindowFlags flags = widget->windowFlags() & ~Qt::WindowType_Mask;
      widget->setWindowFlags(flags | Qt::SubWindow);
      // callbacks
      dockWidget->installEventFilter(this);
      connect(dockWidget, SIGNAL(topLevelChanged(bool)),
              this, SLOT(toplevelChanged(bool)) );
      // show
      computeWindowTitle();
      d->q->addDockWidget(Qt::RightDockWidgetArea, dockWidget);
      ide->loadWindowGeometry(w);
      d->guiUpdateLater();
      widget->show();
      dockWidget->show();
      // set focus
      d->setActive(this);
      QWidget *fw = widget;
      fw = fw->focusWidget() ? fw->focusWidget() : fw;
      fw->setFocus(Qt::OtherFocusReason);
    }
}


void
QLuaMdiMain::Client::toplevelChanged(bool b)
{
  if (dockWidget)
    {
      QDockWidget::DockWidgetFeatures flags = dockWidget->features();
      flags &= ~QDockWidget::DockWidgetVerticalTitleBar;
      if (! b)
        flags |= QDockWidget::DockWidgetVerticalTitleBar;
      dockWidget->setFeatures(flags);
    }
}


#ifdef QT_STEAL_STATUSBAR
static QStatusBar*
takeStatusBar(QMainWindow *mw)
{
  // Making strong assumptions about mainwindow layout.
  QLayout *layout = mw->layout();
  QLayoutItem *item = 0;
  QStatusBar *sb = 0;
  if (layout)
    for (int i=0; (item = layout->itemAt(i)); i++)
      if ((sb = qobject_cast<QStatusBar*>(item->widget())))
        {
          item = layout->takeAt(i);
          sb->hide();
          sb->setParent(0);
          delete item;
          return sb;
        }
  return sb;
}
#endif

void 
QLuaMdiMain::Client::deActivate()
{
  // statusbar
#ifdef QT_STEAL_STATUSBAR
  takeStatusBar(d->q);
  QStatusBar *sb = d->q->statusBar();
  d->q->setStatusBar(sb);
  sb->show();
#endif
}


void 
QLuaMdiMain::Client::reActivate()
{
  // statusbar
#ifdef QT_STEAL_STATUSBAR
  takeStatusBar(d->q);
  QStatusBar *sb = d->q->statusBar();
  if (statusBar)
    sb = statusBar;
  d->q->setStatusBar(sb);
  sb->show();
#endif
}


void 
QLuaMdiMain::Client::stealStatusBar()
{
#ifdef QT_STEAL_STATUSBAR
  QLuaMainWindow *main = qobject_cast<QLuaMainWindow*>(widget);
  if (main)
    statusBar = takeStatusBar(main);
#endif
}


void 
QLuaMdiMain::Client::stealMenuBar()
{
  QMainWindow *main = qobject_cast<QMainWindow*>(widget);
  if (main)
    {
      QMenuBar *mb = main->menuBar();
      if (mb && mb != menuBar)
        {
          menuBar = mb;
          mb->hide();
          mb->installEventFilter(this);
          d->guiUpdateLater();
        }
    }
}


bool
QLuaMdiMain::Client::copyMenuBar(QMenuBar *into)
{
  QMenuBar *orig = menuBar;
  if (!orig)
    return false;
  orig->hide();
  if (orig->actions().isEmpty())
    return false;
  foreach (QAction *action, orig->actions())
    into->addAction(action);
  return true;
}





// ========================================
// QLUAMDIMAIN



QLuaMdiMain::~QLuaMdiMain()
{
  d->closingDown = true;
  while (d->clients.size())
    delete d->clients.takeFirst();
}


QLuaMdiMain::QLuaMdiMain(QWidget *parent)
  : QLuaMainWindow("qLuaMdiMain", parent),
    d(new Private(this))
{
  installEventFilter(d);
  setCentralWidget(d->area);
  setWindowTitle(QApplication::applicationName());
  setDockOptions(QMainWindow::AnimatedDocks);
  menuBar();
#ifdef QT_STEAL_STATUSBAR
  statusBar();
#endif
  loadSettings();
  setClientClass("QWidget");
  d->guiUpdate();
}


QMdiArea* 
QLuaMdiMain::mdiArea()
{
  return d->area;
}


bool
QLuaMdiMain::isActive(QWidget *w)
{
  if (! isActiveWindow())
    return w->isActiveWindow();
  while (w && w->windowType() != Qt::Window)
    {
      if (d->table.contains(w))
        if (d->table[w] == d->active)
          return true;
      w = w->parentWidget();
    }
  return false;
}


bool 
QLuaMdiMain::canClose()
{
  QLuaIde *ide = QLuaIde::instance();
  // close everything if we contain sdimain.
  foreach(Client *c, d->clients)
    if (qobject_cast<QLuaSdiMain*>(c->widget))
      if (ide && !ide->quit(this))
        return false;
  // otherwise close only our windows
  foreach(Client *c, d->clients)
    if (c->widget && !c->widget->close())
      return false;
  // done
  return true;
}


void 
QLuaMdiMain::loadSettings()
{
  QSettings s;
  restoreGeometry(s.value("ide/geometry").toByteArray());
}


void 
QLuaMdiMain::saveMdiSettings()
{
  QSettings s;
  s.setValue("ide/geometry", saveGeometry());
  s.setValue("ide/state", saveState());
  QStringList docked;
  foreach (Client *client, d->clients)
    if (client->widget && client->dockWidget)
      docked += client->widget->objectName();
  s.setValue("ide/dockedWindows", docked);
}


QMenuBar*
QLuaMdiMain::menuBar()
{
#ifdef QT_MAC_USE_NATIVE_MENUBAR
  return QLuaIde::instance()->globalMenuBar();
#else
  return QLuaMainWindow::menuBar();
#endif
}


QMenuBar*
QLuaMdiMain::createMenuBar()
{
  QMenuBar *menubar = new QMenuBar(this);
  QLuaIde::instance()->defaultMenuBar(menubar);
  return menubar;
}


QToolBar *
QLuaMdiMain::createToolBar()
{
  return 0;
}


QStatusBar *
QLuaMdiMain::createStatusBar()
{
  return new QStatusBar(this);
}


bool
QLuaMdiMain::activate(QWidget *w)
{
  Client *client = 0;
  QWidget *window = w->window();
  while (w && w->windowType() != Qt::Window 
         && !d->table.contains(w))
    w = w->parentWidget();
  if (d->table.contains(w))
    {
      window->show();
      window->raise();
      window->activateWindow();
      d->guiUpdateLater();
      Client *client = d->table[w];
      if (client->subWindow)
        client->subWindow->raise();
      w = w->focusWidget() ? w->focusWidget() : w;
      w->setFocus(Qt::OtherFocusReason);
      return true;
    }
  return false;
}


bool
QLuaMdiMain::adopt(QWidget *w)
{
  if (w && w != this)
    {
      foreach(Client *c, d->clients)
        if (c->widget == w)
          return true;
      if (!d->clientClass.isEmpty() && 
          !w->inherits(d->clientClass.constData()) )
        return false;
      show();
      Client *client = new Client(w, d);
      if (w->objectName().startsWith("qLuaSdiMain"))
        {
          QSettings s;
          QStringList docked = s.value("ide/dockedWindows").toStringList();
          if (docked.contains(w->objectName()))
            {
              client->setupAsDockWidget();
              QApplication::sendPostedEvents();
              restoreState(s.value("ide/state").toByteArray());
              return true;
            }
        }
      client->setupAsSubWindow();
      client->subWindow->showNormal();
      return true;
    }
  return false;
}


void 
QLuaMdiMain::adoptAll()
{
  // disown windows that do not belong
  QList<Client*> todel;
  if (!d->clientClass.isEmpty())
    foreach(Client *c, d->clients)
      if (c->widget && !c->widget->inherits(d->clientClass.constData()))
        todel += c;
  foreach(Client *c, todel)
    delete c;
  // adopt windows that belong
  QLuaIde *ide = QLuaIde::instance();
  foreach(QObject *o, ide->windows())
    adopt(qobject_cast<QWidget*>(o));
}


QAction *
QLuaMdiMain::tileAction()
{
  if (! d->tileAction)
    {
      d->tileAction = new QAction(tr("&Tile Windows"), this);
      connect(d->tileAction, SIGNAL(triggered(bool)), 
              d->area, SLOT(tileSubWindows()) );
    }
  return d->tileAction;
}


QAction *
QLuaMdiMain::cascadeAction()
{
  if (! d->cascadeAction)
    {
      d->cascadeAction = new QAction(tr("&Cascade Windows"), this);
      connect(d->cascadeAction, SIGNAL(triggered(bool)), 
              d->area, SLOT(cascadeSubWindows()) );
    }
  return d->cascadeAction;
}


QAction *
QLuaMdiMain::dockAction()
{
  if (! d->dockAction)
    {
      QAction *a = new QAction(tr("&Dock Window"), this);
      connect(a, SIGNAL(triggered(bool)), d, SLOT(dockCurrentWindow(bool)));
      a->setCheckable(true);
      d->dockAction =a;
    }
  return d->dockAction;
}


bool 
QLuaMdiMain::isTabMode() const
{
  return d->tabMode;
}


QByteArray 
QLuaMdiMain::clientClass() const
{
  return d->clientClass;
}


QWidget *
QLuaMdiMain::activeWindow() const
{
  if (d->active)
    return d->active->widget;
  return 0;
}


void 
QLuaMdiMain::setTabMode(bool b)
{
  if (b != d->tabMode)
    {
      d->tabMode = b;
      if (b) 
        d->area->setViewMode(QMdiArea::TabbedView);
      else
        d->area->setViewMode(QMdiArea::SubWindowView);
      foreach(Client *c, d->clients)
        c->computeWindowTitle();
    }
}


void
QLuaMdiMain::setClientClass(QByteArray c)
{
  d->clientClass = c;
}


void
QLuaMdiMain::doNew()
{
  QLuaEditor *n = QLuaIde::instance()->editor();
  n->widget()->setEditorMode("lua");
  n->updateActionsLater();
}




// ========================================
// MOC


#include "qluamdimain.moc"





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
