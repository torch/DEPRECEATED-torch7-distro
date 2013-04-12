/* -*- C++ -*- */

#include <QtGlobal>
#include <QAbstractTextDocumentLayout>
#include <QApplication>
#include <QActionGroup>
#include <QCloseEvent>
#include <QDebug>
#include <QDir>
#include <QDialog>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontMetrics>
#include <QFont>
#include <QFontInfo>
#include <QKeyEvent>
#include <QLabel>
#include <QList>
#include <QMap>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPainter>
#include <QPointer>
#include <QPrintDialog>
#include <QPrinter>
#include <QRegExp>
#include <QSettings>
#include <QStatusBar>
#include <QString>
#include <QStringList>
#include <QSyntaxHighlighter>
#include <QTextBlock>
#include <QTextBlockUserData>
#include <QTextEdit>
#include <QTextCharFormat>
#include <QTextCursor>
#include <QTextFrameFormat>
#include <QTextLayout>
#include <QTextOption>
#include <QToolButton>
#include <QTimer>
#include <QToolBar>
#include <QVariant>
#include <QVariantList>
#include <QVBoxLayout>
#include <QWhatsThis>


#include "qluatextedit.h"
#include "qluamainwindow.h"
#include "qluaeditor.h"

#include "qtluaengine.h"
#include "qluaapplication.h"
#include "qluaconsole.h"
#include "qluaide.h"

using namespace QLuaActionHelpers;


// ========================================
// QLUAEDITOR PRIVATE



class QLuaEditor::Private : public QObject
{
  Q_OBJECT
public:
  ~Private();
  Private(QLuaEditor *q);
public slots:
  void setFileName(QString fname);
  void computeAutoMode();
  void luaEnableActions(bool);
  void luaAcceptingCommands(bool);
  bool luaLoad(bool);
  void updateMode(QLuaTextEditModeFactory *f);
  void doMode(QAction *action);
public:
  QLuaEditor *q;
  QLuaTextEdit *e;
  QPrinter *printer;
  QString fileName;
  QPointer<QPrintDialog> printDialog;
  QPointer<QDialog> gotoDialog;
  QPointer<QDialog> findDialog;
  QPointer<QDialog> replaceDialog;
  QPointer<QActionGroup> modeGroup;
  QLabel *sbPosition;
  QLabel *sbMode;
};


QLuaEditor::Private::~Private()
{
  delete printer;
  printer = 0;
}


QLuaEditor::Private::Private(QLuaEditor *q) 
  : QObject(q), 
    q(q), 
    e(new QLuaTextEdit),
    printer(0),
    sbPosition(0),
    sbMode(0)
{
  e = new QLuaTextEdit(q);
}


void 
QLuaEditor::Private::computeAutoMode()
{
  QString suffix = QFileInfo(fileName).suffix();
  QString firstLine = e->document()->begin().text();
  QRegExp re("-\\*-\\s(\\S+)\\s+-\\*-");
  bool ok = false;
  if (! ok && re.indexIn(firstLine) >= 0)
    ok = e->setEditorMode(re.cap(1));
  if (! ok && ! suffix.isEmpty())
    ok = e->setEditorMode(suffix);
  if (! ok)
    ok = e->setEditorMode(0);
  q->updateActionsLater();
}


void 
QLuaEditor::Private::setFileName(QString fname)
{
  fileName = fname;
  q->setWindowModified(false);
  e->document()->setModified(false);
  e->setDocumentTitle(QFileInfo(fname).fileName());
  QLuaIde::instance()->addRecentFile(fname);
  q->updateActionsLater();
}


void 
QLuaEditor::Private::luaEnableActions(bool enabled)
{
  QLuaTextEditMode *mode = e->editorMode();
  enabled = enabled && mode && mode->supportsLua();
  if (q->hasAction("ActionLuaEval"))
    q->stdAction("ActionLuaEval")->setEnabled(enabled);
  if (q->hasAction("ActionLuaLoad"))
    q->stdAction("ActionLuaLoad")->setEnabled(enabled);
  if (q->hasAction("ActionLuaRestart"))
    q->stdAction("ActionLuaRestart")->setEnabled(enabled);
}


void 
QLuaEditor::Private::luaAcceptingCommands(bool accepting)
{
  luaEnableActions(accepting);
}


bool
QLuaEditor::Private::luaLoad(bool restart)
{
  // check
  if (e->document()->isEmpty())
    return true;
  // call function qtide.doeditor which does the smart thing.
  // but decorate with an informative comment.
  QByteArray cmd;
  if (fileName.isEmpty() || e->document()->isModified())
    {
      cmd = "qtide.doeditor(qt." + q->objectName().toLocal8Bit() + ")";
      if (restart)
        cmd = "require 'qtide'; " + cmd;
    }
  else
    {
      cmd = "dofile('";
      QByteArray f = fileName.toLocal8Bit();
      for (int i=0; i<f.size(); i++)
        if (isascii(f[i]) && isprint(f[i]) && f[i]!='\"' && f[i]!='\'')
          cmd += f[i];
        else {
          char buf[8];
          sprintf(buf,"\\%03o", (unsigned char)f[i]);
          cmd += buf;
        }
      cmd += "\')";
    }
  // eval
  if (restart)
    return QLuaIde::instance()->luaRestart(cmd);
  else
    return QLuaIde::instance()->luaExecute(cmd);
}


void 
QLuaEditor::Private::updateMode(QLuaTextEditModeFactory *f)
{
  if  (modeGroup)
    foreach(QAction *action, modeGroup->actions())
      action->setChecked(qVariantValue<void*>(action->data()) == (void*)f);
}


void
QLuaEditor::Private::doMode(QAction *action)
{
  if (action)
    {
      void *data = qVariantValue<void*>(action->data());
      QLuaTextEditModeFactory *f = (data) ? (QLuaTextEditModeFactory*)data : 0;
      q->doMode(f);
    }
}





// ========================================
// QLUAEDITOR



QLuaEditor::QLuaEditor(QWidget *parent)
  : QLuaMainWindow("editor1", parent), d(new Private(this))
{
  // layout
  setCentralWidget(d->e);
  setFocusProxy(d->e);
  menuBar();
  toolBar();
  statusBar();
  // load settings
  loadSettings();
  // connections
  d->e->setUndoRedoEnabled(true);
  stdAction("ActionEditUndo")->setEnabled(false);
  stdAction("ActionEditRedo")->setEnabled(false);
  connect(d->e, SIGNAL(undoAvailable(bool)), 
          stdAction("ActionEditUndo"), SLOT(setEnabled(bool)) );
  connect(d->e, SIGNAL(redoAvailable(bool)), 
          stdAction("ActionEditRedo"), SLOT(setEnabled(bool)) );
  connect(d->e, SIGNAL(settingsChanged()),
          this, SLOT(updateActionsLater()) );
  connect(d->e, SIGNAL(selectionChanged()),
          this, SLOT(updateActionsLater()) );
  connect(d->e, SIGNAL(textChanged()),
          this, SLOT(updateActionsLater()) );
  connect(d->e, SIGNAL(cursorPositionChanged()),
          this, SLOT(updateStatusBar()) );
  connect(d->e, SIGNAL(cursorPositionChanged()),
          this, SLOT(clearStatusMessage()) );
  connect(QLuaApplication::instance(), SIGNAL(acceptingCommands(bool)),
          d, SLOT(luaAcceptingCommands(bool)) );
  // update actions
  updateActions();
}


void
QLuaEditor::loadSettings()
{
  QSettings s;
  QLuaTextEdit *e = d->e;
  
  // Font
  QFont font = QApplication::font();
  if (s.contains("editor/font"))
    font = qvariant_cast<QFont>(s.value("editor/font"));
  else
    {
      if (! QFontInfo(font).fixedPitch())
        font.setStyleHint(QFont::TypeWriter);
      if (! QFontInfo(font).fixedPitch())
        font.setFamily("Monaco");
      if (! QFontInfo(font).fixedPitch())
        font.setFamily("Courier New");
      if (! QFontInfo(font).fixedPitch())
        font.setFamily("Courier");
      if (! QFontInfo(font).fixedPitch())
        font.setFamily(QString::null);
    }
  e->setFont(font);
  
  // Editor size
  QSize size;
  if (s.contains("editor/size"))
    size = qvariant_cast<QSize>(s.value("editor/size"));
  if (size.width() < 40 || size.width() > 256)
    size.setWidth(80);
  if (size.height() < 10 || size.height() > 256)
    size.setHeight(25);
  e->setSizeInChars(size);

  // Tab size
  int tabSize = -1;
  if (s.contains("editor/tabSize"))
    tabSize = s.value("editor/tabSize").toInt();
  if (tabSize<2 || tabSize>16)
    tabSize = 8;
  e->setTabSize(tabSize);
  
  // Other 
  e->setTabExpand(s.value("editor/tabExpand", true).toBool());
  e->setAutoComplete(s.value("editor/autoComplete", true).toBool());
  e->setAutoIndent(s.value("editor/autoIndent", true).toBool());
  e->setAutoMatch(s.value("editor/autoMatch", true).toBool());
  e->setAutoHighlight(s.value("editor/autoHighlight", true).toBool());
  e->setShowLineNumbers(s.value("editor/showLineNumbers", true).toBool());
  e->setLineWrapMode(s.value("editor/lineWrap",true).toBool() ?
                     QLuaTextEdit::WidgetWidth : QLuaTextEdit::NoWrap);

  // Inherit
  QLuaMainWindow::loadSettings();
}


void
QLuaEditor::saveSettings()
{
  QLuaMainWindow::saveSettings();
  QSettings s;
  QLuaTextEdit *e = d->e;
  s.setValue("editor/lineWrap", e->lineWrapMode() != QLuaTextEdit::NoWrap);
  s.setValue("editor/showLineNumbers", e->showLineNumbers());
  s.setValue("editor/autoComplete", e->autoComplete());
  s.setValue("editor/autoIndent", e->autoIndent());
  s.setValue("editor/autoMatch", e->autoMatch());
  s.setValue("editor/autoHighlight", e->autoHighlight());
}


QString 
QLuaEditor::fileName() const
{
  return d->fileName;
}


void 
QLuaEditor::setFileName(QString fileName)
{
  if (fileName != d->fileName)
    readFile(fileName);
}


QLuaTextEdit *
QLuaEditor::widget()
{
  return d->e;
}


bool 
QLuaEditor::readFile(QFile &file)
{
  if (d->e->readFile(file))
    {
      d->setFileName(QFileInfo(file).canonicalFilePath());
      if (! d->fileName.isEmpty())
        d->computeAutoMode();
      updateActionsLater();
      return true;
    }
  QString an = QCoreApplication::applicationName();
  QString ms = tr("<html>Cannot load file \"%1\".&nbsp;&nbsp;"
                  "<br>%2.</html>")
    .arg(QFileInfo(file).fileName())
    .arg(file.errorString());
  QMessageBox::critical(this, tr("%1 Editor - Error").arg(an), ms);
  return false;
}


bool 
QLuaEditor::readFile(QString fname)
{
  QFile file(fname);
  return readFile(file);
}


bool 
QLuaEditor::writeFile(QFile &file, bool rename)
{
  if (d->e->writeFile(file))
    {
      if (! rename)
        return true;
      d->setFileName(QFileInfo(file).canonicalFilePath());
      updateActionsLater();
      return true;
    }
  QString an = QCoreApplication::applicationName();
  QString ms = tr("<html>Cannot save editor file \"%1\".&nbsp;&nbsp;"
                  "<br>%2.</html>")
    .arg(QFileInfo(file).fileName())
    .arg(file.errorString());
  QMessageBox::critical(this, tr("%1 Editor - Error").arg(an), ms);
  return false;
}


bool 
QLuaEditor::writeFile(QString fname, bool rename)
{
  QFile file(fname);
  return writeFile(file, rename);
}



QAction *
QLuaEditor::createAction(QByteArray name)
{
  if (name == "MenuFile")
    {
      QMenu *menu = newMenu(tr("&File", "file|"));
      menu->addAction(stdAction("ActionFileNew"));
      menu->addAction(stdAction("ActionFileOpen"));
      menu->addAction(stdAction("MenuOpenRecent"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionFileSave"));
      menu->addAction(stdAction("ActionFileSaveAs"));
      menu->addAction(stdAction("ActionFilePrint"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionPreferences"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionFileClose"));
      menu->addAction(stdAction("ActionFileQuit"));
      return menu->menuAction();
    } 
  else if (name == "MenuEdit")
    {
      QMenu *menu = newMenu(tr("&Edit", "edit|"));
      menu->addAction(stdAction("ActionEditUndo"));
      menu->addAction(stdAction("ActionEditRedo"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionEditCut"));
      menu->addAction(stdAction("ActionEditCopy"));
      menu->addAction(stdAction("ActionEditPaste"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionModeBalance"));
      menu->addAction(stdAction("ActionEditGoto"));
      menu->addAction(stdAction("ActionEditFind"));
      menu->addAction(stdAction("ActionEditReplace"));
      return menu->menuAction();
    } 
  else if (name == "MenuTools")
    {
      QMenu  *menu = newMenu(tr("&Tools", "tools|"));
      menu->addAction(stdAction("MenuMode"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionLineWrap"));
      menu->addAction(stdAction("ActionLineNumbers"));
      menu->addAction(stdAction("ActionModeComplete"));
      menu->addAction(stdAction("ActionModeAutoIndent"));
      menu->addAction(stdAction("ActionModeAutoHighlight"));
      menu->addAction(stdAction("ActionModeAutoMatch"));
      return menu->menuAction();
    } 
  else if (name == "MenuLua")
    {
      QMenu *menu = newMenu(tr("&Lua", "lua|"));
      menu->addAction(stdAction("ActionLuaEval"));
      menu->addAction(stdAction("ActionLuaLoad"));
      menu->addAction(stdAction("ActionLuaRestart"));
      return menu->menuAction();
    } 
  else if (name == "MenuMode")
    {
      QMenu *menu = newMenu(tr("Mode","tools|mode"));
      d->modeGroup = new QActionGroup(this);
      d->modeGroup->setExclusive(true);
      connect(d->modeGroup, SIGNAL(triggered(QAction*)),
              d, SLOT(doMode(QAction*)));
      foreach(QLuaTextEditModeFactory *mode, 
              QLuaTextEditModeFactory::factories())
        {
          QAction *action = menu->addAction(mode->name());
          action->setStatusTip("Select the named editor mode.");
          action->setCheckable(true);
          action->setData(qVariantFromValue<void*>(mode));
          d->modeGroup->addAction(action);
        }
      QAction *noneAction = menu->addAction("None");
      noneAction->setStatusTip("Cancel all editor mode.");
      noneAction->setCheckable(true);
      noneAction->setChecked(true);
      d->modeGroup->addAction(noneAction);
      QAction *menuAction = menu->menuAction();
      menuAction->setStatusTip(tr("Select the target language."));
      return menuAction;
    }
  // items
  else if (name == "ActionFileSave")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doSave()));
    }
  else if (name == "ActionFileSaveAs")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doSaveAs()));
    }
  else if (name == "ActionFilePrint")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doPrint()));
    }
  else if (name == "ActionEditSelectAll")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doSelectAll()));
    }
  else if (name == "ActionEditUndo")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doUndo()));
    }
  else if (name == "ActionEditRedo")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doRedo()));
    }
  else if (name == "ActionEditCut")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doCut()));
    }
  else if (name == "ActionEditCopy")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doCopy()));
    }
  else if (name == "ActionEditPaste")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doPaste()));
    }
  else if (name == "ActionEditGoto")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doGoto()));
    }
  else if (name == "ActionEditFind")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doFind()));
    }
  else if (name == "ActionEditReplace")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doReplace()));
    }
  else if (name == "ActionLineNumbers")
    {
      return newAction(tr("Show Line &Numbers", "tools|linenumbers"), false)
        << Connection(this, SLOT(doLineNumbers(bool)))
        << tr("Show line numbers.");
    }
  else if (name == "ActionLineWrap")
    {
      return newAction(tr("&Wrap Long Lines", "tools|linewrap"), false)
        << Connection(this, SLOT(doLineWrap(bool)))
        << tr("Toggle line wrapping mode.");
    }
  else if (name == "ActionModeComplete")
    {
      return newAction(tr("Auto Com&pletion", "tools|complete"), true)
        << Connection(this, SLOT(doCompletion(bool)))
        << tr("Toggle automatic completion with TAB key.");
    }
  else if (name == "ActionModeAutoIndent")
    {
      return newAction(tr("Auto &Indent", "tools|autoindent"), true)
        << Connection(this, SLOT(doAutoIndent(bool)))
        << tr("Toggle automatic indentation with TAB and ENTER keys.");
    }
  else if (name == "ActionModeAutoMatch")
    {
      return newAction(tr("Show &Matches", "tools|automatch"), true)
        << Connection(this, SLOT(doAutoMatch(bool)))
        << tr("Toggle display of matching parenthesis or constructs");
    }
  else if (name == "ActionModeAutoHighlight")
    {
      return newAction(tr("&Colorize", "tools|autohighlight"), true)
        << Connection(this, SLOT(doHighlight(bool)))
        << tr("Toggle syntax colorization.");
    }
  else if (name == "ActionModeBalance")
    {
      return newAction(tr("&Balance", "tools|balance"))
        << Connection(this, SLOT(doBalance()))
        << QIcon(":/images/editbalance.png")
        << QKeySequence(tr("Ctrl+B","lua|lastexpr"))
        << tr("Select successive surrounding syntactical construct.");
    }
  else if (name == "ActionLuaEval")
    {
      return newAction(tr("&Eval Lua Expression","lua|eval"))
        << QKeySequence(tr("Ctrl+E","lua|eval"))
        << QKeySequence(tr("Ctrl+Return", "lua|load"))
        << QKeySequence(tr("Ctrl+Enter","lua|load"))
        << QKeySequence(tr("F4","lua|eval"))
        << QIcon(":/images/playerplay.png")
        << Connection(this, SLOT(doEval()))
        << tr("Evaluate the selected Lua expression.");
    }
  else if (name == "ActionLuaLoad")
    {
      return newAction(tr("&Load Lua File","lua|load"))
        << QIcon(":/images/playerload.png")
        << QKeySequence(tr("Ctrl+L","lua|load"))
        << QKeySequence(tr("F5","lua|load"))
        << Connection(this, SLOT(doLoad()))
        << tr("Load the file into the Lua interpreter.");
    }
  else if (name == "ActionLuaRestart")
    {
      return newAction(tr("&Restart and Load File","lua|restart"))
        << QIcon(":/images/playerrestart.png")
        << QKeySequence(tr("Shift+Ctrl+L","lua|restart"))
        << QKeySequence(tr("Shift+F5","lua|restart"))
        << Connection(this, SLOT(doRestart()))
        << tr("Restart the Lua interpreter and load the file.");
    }
  // default
  return QLuaMainWindow::createAction(name);
}



QToolBar *
QLuaEditor::createToolBar()
{
  QToolBar *toolBar = new QToolBar(this);
  toolBar->addAction(stdAction("ActionFileNew"));
  toolBar->addAction(stdAction("ActionFileOpen"));
  toolBar->addAction(stdAction("ActionFileSave"));
  toolBar->addAction(stdAction("ActionFilePrint"));
  toolBar->addSeparator();
  toolBar->addAction(stdAction("ActionEditUndo"));
  toolBar->addAction(stdAction("ActionEditRedo"));
  toolBar->addAction(stdAction("ActionModeBalance"));
  toolBar->addAction(stdAction("ActionEditFind"));
  toolBar->addSeparator();
  toolBar->addAction(stdAction("ActionLuaEval"));
  QToolButton *luaLoadButton = new QToolButton();
  luaLoadButton->setDefaultAction(stdAction("ActionLuaLoad"));
  luaLoadButton->addAction(stdAction("ActionLuaRestart"));
  toolBar->addWidget(luaLoadButton);
  if (! hasAction("ActionWhatsThis"))
    return toolBar;
  toolBar->addSeparator();
  toolBar->addAction(stdAction("ActionWhatsThis"));
  return toolBar;
}

QMenuBar *
QLuaEditor::createMenuBar()
{
  QMenuBar *menubar = new QMenuBar(this);
  menubar->addAction(stdAction("MenuFile"));
  menubar->addAction(stdAction("MenuEdit"));
  menubar->addAction(stdAction("MenuTools"));
  menubar->addAction(stdAction("MenuLua"));
  menubar->addAction(stdAction("MenuWindows"));
  menubar->addAction(stdAction("MenuHelp"));
  return menubar;
}


QStatusBar *
QLuaEditor::createStatusBar()
{
  QFont font = QApplication::font();
  QFontMetrics metric(font);
  d->sbMode = new QLabel();
  d->sbMode->setFont(font);
  d->sbMode->setAlignment(Qt::AlignCenter);
  d->sbMode->setMinimumWidth(metric.width(" XXXX "));
  d->sbPosition = new QLabel();
  d->sbPosition->setFont(font);
  d->sbPosition->setAlignment(Qt::AlignCenter);
  d->sbPosition->setMinimumWidth(metric.width(" L000 C00 "));
  QStatusBar *sb = new QStatusBar(this);
  sb->addPermanentWidget(d->sbPosition);
  sb->addPermanentWidget(d->sbMode);
  return sb;
}


bool 
QLuaEditor::openFile(QString fname, bool inother)
{
  saveSettings();
  QWidget *w = this;
  if (!inother)
    {
      if (d->fileName.isEmpty() && !d->e->document()->isModified())
        w = 0;
#ifndef Q_WS_MAC
      while (w && !w->inherits("QLuaMdiMain"))
        w = w->parentWidget();
#endif
    }
  if (w)
    QLuaIde::instance()->editor(fname);
  else if (canClose())
    readFile(fname); 
  return true;
}


bool 
QLuaEditor::newDocument()
{
  saveSettings();
  QLuaEditor *e = QLuaIde::instance()->editor();
  QLuaTextEditMode *mode = widget()->editorMode();
  e->widget()->setEditorMode(mode ? mode->factory() : 0);
  e->updateActionsLater();
  return true;
}


void 
QLuaEditor::doSave()
{
  if (d->fileName.isEmpty())
    doSaveAs();
  else if (d->e->document()->isModified())
    writeFile(d->fileName);
}


void 
QLuaEditor::doSaveAs()
{
  QString msg = tr("Save File");
  QString dir = d->fileName;
  QString f = QLuaIde::fileDialogFilters();
  QString s = QLuaIde::allFilesFilter();
  QFileDialog::Options o = QFileDialog::DontUseNativeDialog;
  if (d->e->editorMode())
    s = d->e->editorMode()->filter();
  QString fname = QFileDialog::getSaveFileName(window(), msg, dir, f, &s, o);
  if (! fname.isEmpty())
    writeFile(fname);
}


void 
QLuaEditor::doPrint()
{
  QPrinter *printer = loadPageSetup();
  if (! d->printDialog)
    d->printDialog = new QPrintDialog(printer, this);
  QPrintDialog::PrintDialogOptions options = d->printDialog->enabledOptions();
  options &= ~QPrintDialog::PrintSelection;
  if (d->e->textCursor().hasSelection())
    options |= QPrintDialog::PrintSelection;
  d->printDialog->setEnabledOptions(options);
  if (d->printDialog->exec() == QDialog::Accepted)
    {
      d->e->print(printer);
      savePageSetup();
    }
}


void
QLuaEditor::doSelectAll()
{
  d->e->selectAll();
  updateActionsLater();
}


void
QLuaEditor::doUndo()
{
  d->e->undo();
  updateActionsLater();
}


void
QLuaEditor::doRedo()
{
  d->e->redo();
  updateActionsLater();
}


void
QLuaEditor::doCut()
{
  d->e->cut();
  updateActionsLater();
}


void
QLuaEditor::doCopy()
{
  d->e->copy();
  updateActionsLater();
}


void
QLuaEditor::doPaste()
{
  d->e->paste();
  updateActionsLater();
}


void 
QLuaEditor::doGoto()
{
  QDialog *dialog = d->gotoDialog;
  if (! dialog)
    d->gotoDialog = dialog = d->e->makeGotoDialog();
  d->e->prepareDialog(dialog);
  dialog->exec();
}


void 
QLuaEditor::doFind()
{
  QDialog *dialog = d->findDialog;
  if (! dialog)
    d->findDialog = dialog = d->e->makeFindDialog();
  d->e->prepareDialog(dialog);
  dialog->raise();
  dialog->show();
  dialog->setAttribute(Qt::WA_Moved);
  dialog->activateWindow();
}


void 
QLuaEditor::doReplace()
{
  QDialog *dialog = d->replaceDialog;
  if (! dialog)
    d->replaceDialog = dialog = d->e->makeReplaceDialog();
  d->e->prepareDialog(dialog);
  dialog->raise();
  dialog->show();
  dialog->setAttribute(Qt::WA_Moved);
  dialog->activateWindow();
}


void 
QLuaEditor::doMode(QLuaTextEditModeFactory *factory)
{
  d->e->setEditorMode(factory);
  updateActionsLater();
}


void 
QLuaEditor::doLineNumbers(bool b)
{
  d->e->setShowLineNumbers(b);
  updateActionsLater();
}


void 
QLuaEditor::doLineWrap(bool b)
{
  if (b)
    d->e->setLineWrapMode(QLuaTextEdit::WidgetWidth);
  else
    d->e->setLineWrapMode(QLuaTextEdit::NoWrap);
  updateActionsLater();
}


void 
QLuaEditor::doHighlight(bool b)
{
  d->e->setAutoHighlight(b);
  updateActionsLater();
}


void 
QLuaEditor::doCompletion(bool b)
{
  d->e->setAutoComplete(b);
  updateActionsLater();
}


void
QLuaEditor::doAutoIndent(bool b) 
{
  d->e->setAutoIndent(b);
  updateActionsLater();
}


void
QLuaEditor::doAutoMatch(bool b) 
{
  d->e->setAutoMatch(b);
  updateActionsLater();
}


void 
QLuaEditor::doBalance()
{
  QLuaTextEditMode *mode = d->e->editorMode();
  if (mode && mode->supportsBalance() && mode->doBalance())
    return;
  showStatusMessage(tr("Cannot find enclosing expression."), 5000);
  QLuaApplication::beep();
}


void 
QLuaEditor::doLoad()
{
  QLuaTextEditMode *mode = d->e->editorMode();
  if (mode && mode->supportsLua())
    if (d->luaLoad(false))
      {
        QLuaIde::instance()->activateConsole(this);
        return;
      }
  showStatusMessage(tr("Unable to load file (busy)."), 5000);
  QLuaApplication::beep();
}


void 
QLuaEditor::doRestart()
{
  QLuaTextEditMode *mode = d->e->editorMode();
  if (mode && mode->supportsLua())
    if (d->luaLoad(true))
      {
        QLuaIde::instance()->activateConsole(this);
        return;
      }
  showStatusMessage(tr("Unable to load file (busy)."), 5000);
  QLuaApplication::beep();
}


void 
QLuaEditor::doEval()
{
  QString s;
  QLuaTextEditMode *mode = d->e->editorMode();
  if (mode && mode->supportsLua())
    {
      QTextCursor cursor = d->e->textCursor();
      if (mode->supportsBalance() && !cursor.hasSelection())
        {
          int epos = cursor.position();
          while (mode->doBalance() &&
                 d->e->textCursor().selectionEnd() <= epos)
            cursor = d->e->textCursor();
          d->e->setTextCursor(cursor);
          return;
        }
      if (cursor.hasSelection())
        {
          QString s = cursor.selectedText().trimmed();
          s = s.replace(QChar(0x2029),QChar('\n'));
          if (s.simplified().isEmpty() ||
              QLuaIde::instance()->luaExecute(s.toLocal8Bit()))
            {
              QLuaIde::instance()->activateConsole(this);
              return;
            }
        }
    }
  showStatusMessage(tr("Unable to load file (busy)."), 5000);
  QLuaApplication::beep();
}


void
QLuaEditor::updateStatusBar()
{
  // position
  QTextCursor cursor = d->e->textCursor();
  int line = cursor.blockNumber() + 1;
  QTextBlock block = cursor.block();
  int column = d->e->indentAt(cursor.position(), block);
  d->sbPosition->setText(tr(" L%1 C%2 ").arg(line).arg(column));
  // mode
  QStringList modes;
  if (d->e->editorMode())
    modes += d->e->editorMode()->name();
  if (d->e->lineWrapMode() != QLuaTextEdit::NoWrap)
    modes += tr("Wrap", "Line wrap mode indicator");
  if (d->e->overwriteMode())
    modes += tr("Ovrw", "Overwrite mode indicator");
  d->sbMode->setText(" " + modes.join(" ") + " ");
}


void 
QLuaEditor::updateWindowTitle()
{
  QString fName;
  QString appName = QCoreApplication::applicationName();
  if (! d->fileName.isEmpty())
    fName = QFileInfo(d->fileName).fileName();
#if QT_VERSION >= 0x40400
  setWindowFilePath(fName);
#endif
  if (fName.isEmpty())
    fName = tr("Untitled");
  QString wName = tr("%1[*] - %2 Editor").arg(fName).arg(appName);
  if (windowTitle() != wName)
    setWindowTitle(wName);
  bool modified = d->e->document()->isModified();
  setWindowModified(modified);
}


void 
QLuaEditor::updateActions()
{
  QLuaMainWindow::updateActions();

  // misc
  updateWindowTitle();
  updateStatusBar();

  // cut/paste
  bool readOnly = d->e->isReadOnly();
  bool canPaste = d->e->canPaste();
  bool canCut = d->e->textCursor().hasSelection();
  if (hasAction("ActionEditPaste"))
    stdAction("ActionEditPaste")->setEnabled(canPaste && !readOnly);
  if (hasAction("ActionEditCut"))
    stdAction("ActionEditCut")->setEnabled(canCut && !readOnly);
  if (hasAction("ActionEditCopy"))
    stdAction("ActionEditCopy")->setEnabled(canCut);

  // tools
  if (hasAction("ActionLineNumbers"))
    stdAction("ActionLineNumbers")->setChecked(d->e->showLineNumbers());
  bool wrap = (d->e->lineWrapMode() != QLuaTextEdit::NoWrap);
  if (hasAction("ActionLineWrap"))
    stdAction("ActionLineWrap")->setChecked(wrap);
  QLuaTextEditMode *mode = d->e->editorMode();
  d->updateMode(mode ? mode->factory() : 0);
  if (hasAction("ActionModeAutoHighlight"))
    {
      QAction *action = stdAction("ActionModeAutoHighlight");
      bool autoHighlight = (mode && mode->supportsHighlight());
      action->setChecked(autoHighlight && d->e->autoHighlight());
      action->setEnabled(autoHighlight);
    }
  if (hasAction("ActionModeAutoMatch"))
    {
      QAction *action = stdAction("ActionModeAutoMatch");
      bool autoMatch = (mode && mode->supportsMatch());
      action->setChecked(autoMatch && d->e->autoMatch());
      action->setEnabled(autoMatch);
    }
  if (hasAction("ActionModeAutoIndent"))
    {
      QAction *action = stdAction("ActionModeAutoIndent");
      bool autoIndent = (mode && mode->supportsIndent());
      action->setChecked(autoIndent && d->e->autoIndent());
      action->setEnabled(autoIndent);
    }
  if (hasAction("ActionModeComplete"))
    {
      QAction *action = stdAction("ActionModeComplete");
      bool autoComplete = (mode && mode->supportsComplete());
      action->setChecked(autoComplete && d->e->autoComplete());
      action->setEnabled(autoComplete);
    }
  if (hasAction("ActionModeBalance"))
    {
      bool balance = (mode && mode->supportsBalance());
      stdAction("ActionModeBalance")->setEnabled(balance);
    }
  
  // lua support
  bool luaVisible = mode && mode->supportsLua();
  if (hasAction("ActionLuaEval"))
    stdAction("ActionLuaEval")->setVisible(luaVisible);
  if (hasAction("ActionLuaLoad"))
    stdAction("ActionLuaLoad")->setVisible(luaVisible);
  if (hasAction("ActionLuaRestart"))
    stdAction("ActionLuaRestart")->setVisible(luaVisible);
  QLuaApplication *app = QLuaApplication::instance();
  d->luaEnableActions(luaVisible && app->isAcceptingCommands());
  if (hasAction("MenuLua"))
    stdAction("MenuLua")->setVisible(luaVisible);
}


bool 
QLuaEditor::canClose()
{
  QTextDocument *doc = d->e->document();
  if (isHidden() || !doc->isModified())
    return true;
  QString f = "Untitled";
  if (! d->fileName.isEmpty())
    f = QFileInfo(d->fileName).fileName();
  QString m = tr("File \"%1\" was modified.\n").arg(f);
  QLuaIde::instance()->activateWidget(this);
  switch( QMessageBox::question
          ( window(), tr("Save modified file"), m,
            QMessageBox::Save|QMessageBox::Discard|QMessageBox::Cancel,
            QMessageBox::Cancel) )
    {
    case QMessageBox::Cancel:
      return false;
    case QMessageBox::Discard:
      setWindowModified(false);
      doc->setModified(false);
      return true;
    case QMessageBox::Save:      
      doSave();
    default:
      break;
    }
  return !doc->isModified();
}







// ========================================
// MOC


#include "qluaeditor.moc"





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
