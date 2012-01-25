/* -*- C++ -*- */

#include <QtGlobal>
#include <QApplication>
#include <QActionGroup>
#include <QCloseEvent>
#include <QDebug>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QKeyEvent>
#include <QMenu>
#include <QMetaObject>
#include <QMenuBar>
#include <QPointer>
#include <QPrinter>
#include <QSettings>
#include <QStatusBar>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QToolBar>
#include <QVariant>
#include <QWhatsThis>

#include "qluaapplication.h"
#include "qluasdimain.h"
#include "qluamdimain.h"
#include "qluamainwindow.h"
#include "qluatextedit.h"
#include "qluaeditor.h"
#include "qluabrowser.h"
#include "qluaide.h"



// ========================================
// QLUAMAINWINDOW


using namespace QLuaActionHelpers;


class QLuaMainWindow::Private : public QObject
{
  Q_OBJECT
public:
  QLuaMainWindow *q;
  QPointer<QMenuBar> menuBar;
  QPointer<QToolBar> toolBar;
  QPointer<QStatusBar> statusBar;
  bool updateActionsScheduled;
  QMap<QByteArray,QAction*> actions;
  QString statusMessage;
  QPrinter *printer;
public:
  ~Private();
  Private(QLuaMainWindow *parent);
public slots:
  void messageChanged(QString);
};


QLuaMainWindow::Private::~Private()
{
  delete printer;
}


QLuaMainWindow::Private::Private(QLuaMainWindow *q)
  : QObject(q), q(q), 
    updateActionsScheduled(false),
    printer(0)
{
}


void 
QLuaMainWindow::Private::messageChanged(QString s)
{
  if (statusBar && s.isEmpty() && !statusMessage.isEmpty())
    statusBar->showMessage(statusMessage);
}


QLuaMainWindow::~QLuaMainWindow()
{
  QLuaIde *ide = QLuaIde::instance();
  ide->saveRecentFiles();
}


QLuaMainWindow::QLuaMainWindow(QString objname, QWidget *parent)
  : QMainWindow(parent), d(new Private(this))
{
  setObjectName(objname);
}


QMenuBar *
QLuaMainWindow::menuBar()
{
  if (! d->menuBar)
    if ((d->menuBar = createMenuBar()))
      setMenuBar(d->menuBar);
  return d->menuBar;
}


QMenuBar *
QLuaMainWindow::createMenuBar()
{
  return 0;
}


QToolBar *
QLuaMainWindow::toolBar()
{
  if (! d->toolBar)
    if ((d->toolBar = createToolBar()))
      {
        if (d->toolBar->objectName().isEmpty())
          d->toolBar->setObjectName("mainToolBar");
        if (d->toolBar->windowTitle().isEmpty())
          d->toolBar->setWindowTitle(tr("Main ToolBar"));
        addToolBar(Qt::TopToolBarArea, d->toolBar);
      }
  return d->toolBar;
}


QToolBar *
QLuaMainWindow::createToolBar()
{
  return 0;
}


QStatusBar *
QLuaMainWindow::statusBar()
{
  if (! d->statusBar)
    if ((d->statusBar = createStatusBar()))
      {
        setStatusBar(d->statusBar);
        connect(d->statusBar, SIGNAL(messageChanged(QString)),
                d, SLOT(messageChanged(QString)) );
      }
  return d->statusBar;
}


QStatusBar *
QLuaMainWindow::createStatusBar()
{
  return 0;
}


QPrinter *
QLuaMainWindow::loadPageSetup()
{
  QPrinter *printer = d->printer;
  if (! printer)
    d->printer = printer = new QPrinter;
  QSettings s;
  if (s.contains("printer/pageSize"))
    {
      int n = s.value("printer/pageSize").toInt();
      if (n >= 0 || n < QPrinter::Custom)
        printer->setPaperSize((QPrinter::PageSize)n);
    }
  if (s.contains("printer/pageMargins/unit"))
    {
      int unit = s.value("printer/pageMargins/unit").toInt();
      if (unit >= 0 && unit < QPrinter::DevicePixel)
        {
          qreal left,top,right,bottom;
          left = s.value("printer/pageMargins/left").toDouble();  
          top = s.value("printer/pageMargins/top").toDouble();  
          right = s.value("printer/pageMargins/right").toDouble();  
          bottom = s.value("printer/pageMargins/bottom").toDouble();
          printer->setPageMargins(left,top,right,bottom,(QPrinter::Unit)unit);
        }
    }
  return printer;
}


void 
QLuaMainWindow::savePageSetup()
{
  QPrinter *printer = d->printer;
  if (printer)
    {
      QSettings s;
      QPrinter::PageSize ps = printer->paperSize();
      if (ps < QPrinter::Custom)
        s.setValue("printer/pageSize", (int)(printer->pageSize()));
      qreal left,top,right,bottom;
      QPrinter::Unit unit = QPrinter::Millimeter;
      printer->getPageMargins(&left,&top,&right,&bottom,unit);
      s.setValue("printer/pageMargins/left", left);  
      s.setValue("printer/pageMargins/top", top);  
      s.setValue("printer/pageMargins/right", right);  
      s.setValue("printer/pageMargins/bottom", bottom);
      s.setValue("printer/pageMargins/unit", (int)unit);
    }
}


void 
QLuaMainWindow::loadSettings()
{
  QLuaIde *ide = QLuaIde::instance();
  ide->loadWindowGeometry(this);
}


void 
QLuaMainWindow::saveSettings()
{
  QLuaIde *ide = QLuaIde::instance();
  ide->saveWindowGeometry(this);
}


QAction *
QLuaMainWindow::newAction(QString text)
{
  QAction *action = new QAction(text, this);
  action->setMenuRole(QAction::NoRole);
  return action;
}


QAction *
QLuaMainWindow::newAction(QString text, bool flag)
{
  QAction *action = new QAction(text, this);
  action->setMenuRole(QAction::NoRole);
  action->setCheckable(true);
  action->setChecked(flag);
  return action;
}


QMenu *
QLuaMainWindow::newMenu(QString title)
{
  QMenu *menu = new QMenu(title, this);
  QAction *action = menu->menuAction();
  action->setMenuRole(QAction::NoRole);
  return menu;
}


QAction *
QLuaMainWindow::hasAction(QByteArray name)
{
  if (d->actions.contains(name))
    return d->actions[name];
  return 0;
}


QAction*
QLuaMainWindow::stdAction(QByteArray name)
{
  if (d->actions.contains(name))
    return d->actions[name];
  QAction *action = createAction(name);
  if (action)
    d->actions[name] = action;
  return action;
}

QAction*
QLuaMainWindow::createAction(QByteArray name)
{
  // defines standard item appearance
  if (name == "ActionFileSave")
    {
      return newAction(tr("&Save", "file|save"))
        << QKeySequence(QKeySequence::Save)
        << QIcon(":/images/filesave.png")
        << tr("Save the contents of this window.");
    }
  else if (name == "ActionFileSaveAs")
    {
      return newAction(tr("Save &As...", "file|saveas"))
        << QIcon(":/images/filesaveas.png")
        << tr("Save the contents of this window into a new file.");
    } 
  else if (name == "ActionFilePrint")
    {
      return newAction(tr("&Print...", "file|print"))
        << QKeySequence(QKeySequence::Print)
        << QIcon(":/images/fileprint.png")
        << tr("Print the contents of this window.");
    }
  else if (name == "ActionEditSelectAll")
    {
      return newAction(tr("Select &All", "edit|selectall"))
        << QKeySequence(QKeySequence::SelectAll)
        << tr("Select everything.");
    }
  else if (name == "ActionEditUndo")
    {
      return newAction(tr("&Undo", "edit|undo"))
        << QKeySequence(QKeySequence::Undo)
        << QIcon(":/images/editundo.png")
        << tr("Undo last edit.");
    }
  else if (name == "ActionEditRedo")
    {
      return newAction(tr("&Redo", "edit|redo"))
        << QKeySequence(QKeySequence::Redo)
        << QIcon(":/images/editredo.png")
        << tr("Redo last undo.");
    }
  else if (name == "ActionEditCut")
    {
      return newAction(tr("Cu&t", "edit|cut"))
        << QKeySequence(QKeySequence::Cut)
        << QIcon(":/images/editcut.png")
        << tr("Cut selection to clipboard.");
    }
  else if (name == "ActionEditCopy")
    {
      return newAction(tr("&Copy", "edit|copy"))
        << QKeySequence(QKeySequence::Copy)
        << QIcon(":/images/editcopy.png")
        << tr("Copy selection to clipboard.");
    }
  else if (name == "ActionEditPaste")
    {
      return newAction(tr("&Paste", "edit|paste"))
        << QKeySequence(QKeySequence::Paste)
        << QIcon(":/images/editpaste.png")
        << tr("Paste from clipboard.");
    }
  else if (name == "ActionEditGoto")
    {
      return newAction(tr("&Go to Line", "edit|goto"))
        << QKeySequence(tr("Ctrl+G", "edit|goto"))
        << QIcon(":/images/editgoto.png")
        << tr("Go to numbered line.");
    }
  else if (name == "ActionEditFind")
    {
      return newAction(tr("&Find", "edit|find"))
        << QKeySequence(QKeySequence::Find)
        << QIcon(":/images/editfind.png")
        << tr("Find text.");
    }
  else if (name == "ActionEditReplace")
    {
      return newAction(tr("&Replace", "edit|findreplace"))
        << QKeySequence(QKeySequence::Replace)
        << QIcon(":/images/editreplace.png")
        << tr("Find and replace text.");
    }
  // default
  return QLuaIde::instance()->stdAction(name);
}


bool 
QLuaMainWindow::canClose()
{
  return true;
}


void
QLuaMainWindow::closeEvent(QCloseEvent *event)
{
  if (isHidden())
    event->accept();
  else if (canClose()) {
    saveSettings();
    event->accept();
  } else 
    event->ignore();
}


void 
QLuaMainWindow::updateActionsLater()
{
  if (! d->updateActionsScheduled)
    {
      d->updateActionsScheduled = true;
      QTimer::singleShot(0, this, SLOT(updateActions()));
    }
}


void 
QLuaMainWindow::updateActions()
{
  d->updateActionsScheduled = false;
}


void 
QLuaMainWindow::clearStatusMessage()
{
  d->statusMessage.clear();
  if (d->statusBar)
    d->statusBar->clearMessage();
}


void 
QLuaMainWindow::showStatusMessage(const QString & message, int timeout)
{
  if (! d->statusBar)
    d->statusBar = QMainWindow::statusBar();
  if (! timeout)
    d->statusMessage = message;
  if (d->statusBar)
    d->statusBar->showMessage(message, timeout);
}


bool 
QLuaMainWindow::openFile(QString fileName, bool inOther)
{
  return false;
}


bool 
QLuaMainWindow::newDocument()
{
  return false;
}




// ========================================
// MOC


#include "qluamainwindow.moc"





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
