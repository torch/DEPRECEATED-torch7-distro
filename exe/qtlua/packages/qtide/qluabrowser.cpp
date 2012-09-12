/* -*- C++ -*- */

#include <QtGlobal>
#include <QApplication>
#include <QActionGroup>
#include <QCloseEvent>
#include <QDebug>
#include <QDesktopServices>
#include <QDialog>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QInputDialog>
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
#include <QProgressBar>
#include <QRegExp>
#include <QSettings>
#include <QShortcut>
#include <QStatusBar>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QToolBar>
#include <QVariant>
#include <QVariantList>
#include <QWhatsThis>

#include "qluamainwindow.h"
#include "qluabrowser.h"
#include "qluatextedit.h"
#include "qluaeditor.h"

#include "qtluaengine.h"
#include "qluaapplication.h"
#include "qluaconsole.h"
#include "qluaide.h"

#if HAVE_QTWEBKIT
# include <QWebSettings>
# include <QNetworkRequest>
#endif

class QNetworkRequest;

using namespace QLuaActionHelpers;


// ========================================
// QLUABROWSER PRIVATE


class QLuaBrowser::Private : public QObject
{
  Q_OBJECT
public:
  ~Private();
  Private(QLuaBrowser *q);
public:
  QLuaBrowser *q;
  QString baseTitle;
  QUrl homeUrl;
  QUrl url;
#if HAVE_QTWEBKIT
  QPointer<QWebView> w;
  QPointer<FindDialog> findDialog;
  QPointer<QPrintDialog> printDialog;
#endif
public slots:
  void loadRequested(const QNetworkRequest &request);
  void loadStarted();
  void loadFinished(bool);
};


QLuaBrowser::Private::~Private()
{
}


QLuaBrowser::Private::Private(QLuaBrowser *q) 
  : QObject(q), q(q)
{
}


void 
QLuaBrowser::Private::loadRequested(const QNetworkRequest &request)
{
#if HAVE_QTWEBKIT
  url = request.url();
#endif
}

void 
QLuaBrowser::Private::loadStarted()
{
#if HAVE_QTWEBKIT
  q->statusBar()->showMessage(QString::null);
#endif
}

void 
QLuaBrowser::Private::loadFinished(bool ok)
{
#if HAVE_QTWEBKIT
  if (! ok)
    {
      QString message;
      if (url.isEmpty())
        message = tr("Error while loading requested url.");
      else
        message = tr("Error while loading \"%1\".").arg(url.toString());
      q->statusBar()->showMessage(message);
    }
  url = QUrl();
#endif
}


// ========================================
// QLUABROWSER WEBVIEW


#if HAVE_QTWEBKIT
class QLuaBrowser::WebView : public QWebView
{
public:
  WebView(QWidget *parent) : QWebView(parent) {}
protected:
  virtual QWebView *createWindow(QWebPage::WebWindowType);
};

QWebView *
QLuaBrowser::WebView::createWindow(QWebPage::WebWindowType) 
{
  QLuaBrowser *e = QLuaIde::instance()->browser();
  return e->d->w;
}
#endif


// ========================================
// FIND DIALOG


#include "ui_qluafinddialog.h"

class QLuaBrowser::FindDialog : public QDialog
{
  Q_OBJECT
protected:
  Ui_QLuaFindDialog ui;
  QLuaBrowser *browser;
  QWebView *view;
  QPointer<QShortcut> findNextSCut;
  QPointer<QShortcut> findPrevSCut;
public:
  FindDialog(QLuaBrowser *browser);
  ~FindDialog();
  void prepare();
protected slots:
  void update();
  bool find(bool);
  void findNext()     { find(false); }
  void findPrevious() { find(true); }
  void next();
};

QLuaBrowser::FindDialog::FindDialog(QLuaBrowser *browser)
  : QDialog(browser), 
    browser(browser), 
    view(0)
{
#if HAVE_QTWEBKIT
  view = browser->d->w;
  // ui
  ui.setupUi(this);
  connect(ui.findButton,SIGNAL(clicked()),this, SLOT(next()));
  connect(ui.findEdit,SIGNAL(textChanged(QString)), this, SLOT(update())); 
  connect(ui.findEdit,SIGNAL(returnPressed()), this, SLOT(next())); 
  new QShortcut(QKeySequence::FindNext, this, SLOT(findNext()));
  new QShortcut(QKeySequence::FindPrevious, this, SLOT(findPrevious()));
  findNextSCut = new QShortcut(QKeySequence::FindNext, browser);
  findPrevSCut = new QShortcut(QKeySequence::FindPrevious, browser);
  connect(findNextSCut, SIGNAL(activated()), this, SLOT(findNext()));
  connect(findPrevSCut, SIGNAL(activated()), this, SLOT(findPrevious()));
  update();
  // settings
  QSettings s;
  bool c = s.value("browser/find/caseSensitive",false).toBool();
  bool w = s.value("browser/find/wholeWords",true).toBool();
  ui.caseSensitiveBox->setChecked(c);
  ui.wholeWordsBox->setChecked(w);
#endif
}

QLuaBrowser::FindDialog::~FindDialog()
{
#if HAVE_QTWEBKIT
  delete findNextSCut;
  delete findPrevSCut;
  QSettings s;
  s.setValue("browser/find/caseSensitive", ui.caseSensitiveBox->isChecked());
  s.setValue("browser/find/wholeWords", ui.wholeWordsBox->isChecked());
#endif
}


void
QLuaBrowser::FindDialog::prepare()
{
#if HAVE_QTWEBKIT
  QString s = view->selectedText();
  if (! s.isEmpty())
    {
      if (s.contains(QChar(0x2029)))
        s.truncate(s.indexOf(QChar(0x2029)));
      if (! s.isEmpty()) 
        ui.findEdit->setText(s);
      ui.findEdit->selectAll();
      ui.findEdit->setFocus();
    }
  update();
#endif
}


void
QLuaBrowser::FindDialog::update()
{
#if HAVE_QTWEBKIT
  ui.findButton->setEnabled(! ui.findEdit->text().isEmpty());
  ui.wholeWordsBox->setChecked(false);
  ui.wholeWordsBox->setEnabled(false);
#endif
}


bool
QLuaBrowser::FindDialog::find(bool backward)
{
#if HAVE_QTWEBKIT
  if (ui.findEdit->text().isEmpty())
    return false;
  QWebPage::FindFlags flags = 0;
  if (backward)
    flags |= QWebPage::FindBackward;
  if (ui.caseSensitiveBox->isChecked())
    flags |= QWebPage::FindCaseSensitively;
  return view->findText(ui.findEdit->text(), flags);
#else
  return false;
#endif
}


void
QLuaBrowser::FindDialog::next()
{
#if HAVE_QTWEBKIT
  if (! find(ui.searchBackwardsBox->isChecked()))
    QMessageBox::warning(this, tr("Find Warning"), 
                         tr("Search text not found."));    
#endif
}


// ========================================
// QLUABROWSER


QLuaBrowser::QLuaBrowser(QWidget *parent)
  : QLuaMainWindow("browser1", parent), d(new Private(this))
{
  // layout
#if HAVE_QTWEBKIT
  d->w = new WebView(this);
  setCentralWidget(d->w);
  setFocusProxy(d->w);
#endif
  menuBar();
  toolBar();
  statusBar();
  // load settings
  loadSettings();
  // connections
#if HAVE_QTWEBKIT
  connect(d->w, SIGNAL(titleChanged(QString)),
          this, SLOT(updateActionsLater()) );
  connect(d->w, SIGNAL(iconChanged()),
          this, SLOT(updateActionsLater()) );
  connect(d->w, SIGNAL(urlChanged(QUrl)),
          this, SLOT(updateActionsLater()) );
  connect(d->w, SIGNAL(selectionChanged()),
          this, SLOT(updateActionsLater()) );
  connect(d->w->page(), SIGNAL(downloadRequested(QNetworkRequest)),
          d, SLOT(loadRequested(QNetworkRequest)) );
  connect(d->w, SIGNAL(loadStarted()),
          d, SLOT(loadStarted()) );
  connect(d->w, SIGNAL(loadFinished(bool)),
          d, SLOT(loadFinished(bool)) );
#endif
  // shortcuts
#if HAVE_QTWEBKIT
  new QShortcut(QKeySequence("Ctrl+0"), this, SLOT(doZoomReset()));
#endif
  // update actions
  updateActions();
}


void
QLuaBrowser::loadSettings()
{
  QLuaMainWindow::loadSettings();
}


void
QLuaBrowser::saveSettings()
{
  QLuaMainWindow::saveSettings();
}


QUrl
QLuaBrowser::url() const
{
#if HAVE_QTWEBKIT
  return d->w->url();
#else
  return d->url;
#endif
}


QUrl
QLuaBrowser::homeUrl() const
{
  return d->homeUrl;
}


void 
QLuaBrowser::setUrl(QUrl url)
{
  d->url = url;
#if HAVE_QTWEBKIT
  if (url != d->w->url())
    d->w->setUrl(url);
#else
  QDesktopServices::openUrl(url);
#endif
}


void 
QLuaBrowser::setHomeUrl(QUrl url)
{
  d->homeUrl = url;
}


QString
QLuaBrowser::pageTitle() const
{
#if HAVE_QTWEBKIT
  return d->w->title();
#else
  return QString();
#endif
}


QString
QLuaBrowser::baseTitle() const
{
  return d->baseTitle;
}


void
QLuaBrowser::setBaseTitle(QString s)
{
  d->baseTitle = s;
  updateWindowTitle();
}


QString
QLuaBrowser::toHtml() const
{
#if HAVE_QTWEBKIT
  return d->w->page()->mainFrame()->toHtml();
#else
  return QString::null;
#endif
}


void
QLuaBrowser::setHtml(QString s)
{
#if HAVE_QTWEBKIT
  d->w->setHtml(s);
#endif
}


QWebView *
QLuaBrowser::view()
{
#if HAVE_QTWEBKIT
  return d->w;
#else
  return 0;
#endif
}


QWebPage *
QLuaBrowser::page()
{
#if HAVE_QTWEBKIT
  return d->w->page();
#else
  return 0;
#endif
}




QAction *
QLuaBrowser::createAction(QByteArray name)
{
  // menus
#if HAVE_QTWEBKIT
  if (name == "MenuFile")
    {
      QMenu *menu = newMenu(tr("&File", "file|"));
      menu->addAction(stdAction("ActionFileNew"));
      menu->addAction(stdAction("ActionFileOpen"));
      menu->addAction(stdAction("ActionBrowseOpenLocation"));
      menu->addAction(stdAction("MenuOpenRecent"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionBrowseOpenBrowser"));
      menu->addAction(stdAction("ActionFileSaveAs"));
      menu->addAction(stdAction("ActionFilePrint"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionPreferences"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionFileClose"));
      menu->addAction(stdAction("ActionFileQuit"));
      return menu->menuAction();
    }
  // menus
  else if (name == "MenuEdit")
    {
      QMenu *menu = newMenu(tr("&Edit", "edit|"));
      menu->addAction(stdAction("ActionEditCut"));
      menu->addAction(stdAction("ActionEditCopy"));
      menu->addAction(stdAction("ActionEditPaste"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionBrowseEdit"));
      menu->addAction(stdAction("ActionEditFind"));
      return menu->menuAction();
     }
  else if (name == "MenuView")
    {
      QMenu *menu = newMenu(tr("&View","view|"));
      menu->addAction(stdAction("ActionZoomIn"));
      menu->addAction(stdAction("ActionZoomOut"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionBrowseBack"));
      menu->addAction(stdAction("ActionBrowseForward"));
      menu->addAction(stdAction("ActionBrowseHome"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionBrowseReload"));
      return menu->menuAction();
     }
  // items
  else if (name == "ActionFileSaveAs")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doSaveAs()))
        << QIcon(":/images/filesave.png")
        << QKeySequence(QKeySequence::Save);
    }
  else if (name == "ActionFilePrint")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doPrint()));
    }
  else if (name == "ActionEditCut")
    {
      return d->w->pageAction(QWebPage::Cut)
        << NewAction(tr("Cu&t", "edit|cut"))
        << QKeySequence(QKeySequence::Cut)
        << QIcon(":/images/editcut.png")
        << tr("Cut selection to clipboard.");
    }
  else if (name == "ActionEditCopy")
    {
      return d->w->pageAction(QWebPage::Copy)
        << NewAction(tr("&Copy", "edit|copy"))
        << QKeySequence(QKeySequence::Copy)
        << QIcon(":/images/editcopy.png")
        << tr("Copy selection to clipboard.");
    }
  else if (name == "ActionEditPaste")
    {
      return d->w->pageAction(QWebPage::Paste)
        << NewAction(tr("&Paste", "edit|paste"))
        << QKeySequence(QKeySequence::Paste)
        << QIcon(":/images/editpaste.png")
        << tr("Paste from clipboard.");
    }
  else if (name == "ActionEditFind")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doFind()));
    }
  else if (name == "ActionZoomIn")
    {
      return newAction("Zoom In")
        << QKeySequence("Ctrl++")
        << Connection(this, SLOT(doZoomIn()))
        << tr("Enlarge text and images");
    }
  else if (name == "ActionZoomOut")
    {
      return newAction("Zoom Out")
        << QKeySequence("Ctrl+-")
        << Connection(this, SLOT(doZoomOut()))
        << tr("Shrink text and images");
    }
  else if (name == "ActionBrowseBack")
    {
      QAction *a = d->w->pageAction(QWebPage::Back);
      return newAction(a->text())
        << a->statusTip() << a->whatsThis() << a->icon()
        << QKeySequence(QKeySequence::Back)
        << Connection(this, SLOT(doBackward()));
    }
  else if (name == "ActionBrowseForward")
    {
      QAction *a = d->w->pageAction(QWebPage::Forward);
      return newAction(a->text())
        << a->statusTip() << a->whatsThis() << a->icon()
        << QKeySequence(QKeySequence::Forward)
        << Connection(this, SLOT(doForward()));
    }
  else if (name == "ActionBrowseReload")
    {
      return d->w->pageAction(QWebPage::Reload)
        << QKeySequence(QKeySequence::Refresh);
    }
  else if (name == "ActionBrowseStop")
    {
      return d->w->pageAction(QWebPage::Stop);
    }
  else if (name == "ActionBrowseHome")
    {
      return newAction(tr("&Home", "browse|home"))
        << QIcon(":/images/home.png")
        << QKeySequence(tr("Ctrl+Home", "browse|home"))
        << Connection(this, SLOT(doHome()))
        << tr("Return to the home page.");
    }
  else if (name == "ActionBrowseEdit")
    {
      return newAction(tr("&Edit HTML", "browse|edit"))
        << QIcon(":/images/filenew.png")
        << QKeySequence(tr("Ctrl+E", "browse|edit"))
        << Connection(this, SLOT(doEdit()))
        << tr("Edit the HTML code for this page.");
    }
  else if (name == "ActionBrowseOpenLocation")
    {
      return newAction(tr("Open &Location...", "browse|openlocation"))
        << QIcon(":/images/web.png")
        << QKeySequence(tr("Ctrl+L", "browse|openlocation"))
        << Connection(this, SLOT(doOpenLocation()))
        << tr("Open an arbitrary web location.");
    }
  else if (name == "ActionBrowseOpenBrowser")
    {
      return newAction(tr("Open in External &Browser", "browse|openbrowser"))
        << QIcon(":/images/browser.png")
        << QKeySequence(tr("Ctrl+B", "browse|openbrowser"))
        << Connection(this, SLOT(doOpenBrowser()))
        << tr("Open this page with the default system browser.");
    }
#endif
  return QLuaMainWindow::createAction(name);
}



QMenuBar *
QLuaBrowser::createMenuBar()
{
#if HAVE_QTWEBKIT
  QMenuBar *menubar = new QMenuBar(this);
  menubar->addAction(stdAction("MenuFile"));
  menubar->addAction(stdAction("MenuEdit"));
  menubar->addAction(stdAction("MenuView"));
  menubar->addAction(stdAction("MenuWindows"));
  menubar->addAction(stdAction("MenuHelp"));
  return menubar;
#else
  return QLuaMainWindow::createMenuBar();
#endif
}


QToolBar *
QLuaBrowser::createToolBar()
{
#if HAVE_QTWEBKIT
  QToolBar *toolBar = new QToolBar(this);
  toolBar->addAction(stdAction("ActionBrowseBack"));
  toolBar->addAction(stdAction("ActionBrowseForward"));
  toolBar->addAction(stdAction("ActionBrowseHome"));
  toolBar->addSeparator();
  toolBar->addAction(stdAction("ActionBrowseReload"));
  toolBar->addAction(stdAction("ActionBrowseStop"));
  return toolBar;
#else
  return QLuaMainWindow::createToolBar();
#endif
}


QStatusBar *
QLuaBrowser::createStatusBar()
{
#if HAVE_QTWEBKIT
  QStatusBar *statusBar = new QStatusBar(this);
  QProgressBar *progressBar = new QProgressBar(statusBar);
  progressBar->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Ignored);
  statusBar->addWidget(progressBar);
  connect(d->w, SIGNAL(loadStarted()), 
          progressBar, SLOT(show()));
  connect(d->w, SIGNAL(loadProgress(int)), 
          progressBar, SLOT(setValue(int)));
  connect(d->w, SIGNAL(loadFinished(bool)), 
          progressBar, SLOT(hide()));
  connect(d->w, SIGNAL(statusBarMessage(QString)),
          statusBar, SLOT(showMessage(QString)) );
  connect(d->w->page(), SIGNAL(linkHovered(QString,QString,QString)),
          statusBar, SLOT(showMessage(QString)) );
  return statusBar;
#else
  return QLuaMainWindow::createStatusBar();
#endif
}


void 
QLuaBrowser::updateWindowTitle()
{
  QString b = baseTitle();
  QString p = pageTitle();
  if (b.isEmpty())
    b = tr("QLua Browser");
  if (p.isEmpty())
    b = tr("%1[*]").arg(b);
  else
    b = tr("%1[*] - %2").arg(p).arg(b);
  if (windowTitle() != b)
    setWindowTitle(b);
#if QT_VERSION >= 0x40400
  setWindowFilePath(d->url.toLocalFile());
#endif
}


void 
QLuaBrowser::updateWindowIcon()
{
#if HAVE_QTWEBKIT
  QIcon icon = d->w->icon();
  if (icon.isNull())
    setWindowIcon(QApplication::windowIcon());
  else
    setWindowIcon(icon);    
#endif
}


void 
QLuaBrowser::updateActions()
{
#if HAVE_QTWEBKIT
  updateWindowIcon();
  updateWindowTitle();
  d->url = d->w->url();
  bool validUrl = d->url.isValid();
  stdAction("ActionBrowseEdit")->setEnabled(validUrl);
  stdAction("ActionBrowseOpenBrowser")->setEnabled(validUrl);
#endif
  QLuaMainWindow::updateActions();
}


bool 
QLuaBrowser::openFile(QString fname, bool inother)
{
  saveSettings();
  // do we open in a new window?
  QWidget *w = this;
  if (!inother)
    {
      if (url().isEmpty())
        w = 0;
#ifndef Q_WS_MAC
      while (w && !w->inherits("QLuaMdiMain"))
        w = w->parentWidget();
#endif
    }
  // do we open as html?
  bool openInEditor = false;
  bool openInBrowser = false;
  QString suffix = QFileInfo(fname).suffix();
  if (suffix == "html" || suffix == "HTML")
    openInBrowser = true;
  foreach(QLuaTextEditModeFactory *mode, 
          QLuaTextEditModeFactory::factories())
    if (mode->suffixes().contains(suffix))
      openInEditor = true;
  // proceed
  if (openInEditor && ! openInBrowser)
    QLuaIde::instance()->editor(fname);
  else if (w)
    QLuaIde::instance()->browser(fname);
  else if (canClose())
    setUrl(fname);
  return true;
}


bool 
QLuaBrowser::newDocument()
{
  QLuaIde::instance()->browser();
  return true;
}


void 
QLuaBrowser::doOpenLocation()
{
#if HAVE_QTWEBKIT
  QString c = tr("Open Location", "dialog caption");
  QString l = tr("Please type the URL of the document you want to open.");
  QUrl url = d->w->url();
  QString t = (url.isValid()) ? url.toString() : QString("http://");
  bool ok;
  url  = QInputDialog::getText(this, c, l, QLineEdit::Normal, t, &ok);
  if (ok && url.isValid())
    setUrl(url);
  else if (ok)
    QMessageBox::critical(this, tr("%1 - Error").arg(this->baseTitle()),
                          tr("<html>The URL you entered is invalid</html>"));
#endif  
}


void 
QLuaBrowser::doOpenBrowser()
{
#if HAVE_QTWEBKIT
  QUrl url = d->w->url();
#else
  QUrl url = d->url;
#endif
  if (url.isValid())
    QDesktopServices::openUrl(url);
}


void
QLuaBrowser::doSaveAs()
{
#if HAVE_QTWEBKIT
  QString msg = tr("Save File");
  QString dir = url().toLocalFile();
  QString f = QLuaIde::fileDialogFilters();
  QString s = QLuaIde::htmlFilesFilter();
  QFileDialog::Options o = QFileDialog::DontUseNativeDialog;
  QString fname = QFileDialog::getSaveFileName(window(), msg, dir, f, &s, o);
  if (! fname.isEmpty())
    {
      QFile file(fname);
      if (file.open(QIODevice::WriteOnly))
        {
          QTextStream out(&file);
          QApplication::setOverrideCursor(Qt::WaitCursor);
          out << d->w->page()->mainFrame()->toHtml();
        }
      if (file.error())
        {
          QString title = tr("%1 - Error").arg(this->baseTitle());
          QString ms = 
            tr("<html>Cannot save html file \"%1\".&nbsp;&nbsp;<br>%2.</html>")
            .arg(QFileInfo(file).fileName())
            .arg(file.errorString());
          QMessageBox::critical(this, title, ms);
        }
    }
#endif
}


void
QLuaBrowser::doPrint()
{
#if HAVE_QTWEBKIT
  QPrinter *printer = loadPageSetup();
  if (! d->printDialog)
    d->printDialog = new QPrintDialog(printer, this);
  QPrintDialog::PrintDialogOptions options = d->printDialog->enabledOptions();
  options &= ~QPrintDialog::PrintSelection;
  d->printDialog->setEnabledOptions(options);
  if (d->printDialog->exec() == QDialog::Accepted)
    {
      d->w->print(printer);
      savePageSetup();
    }
#endif
}


void 
QLuaBrowser::doCopy()
{
#if HAVE_QTWEBKIT
  d->w->triggerPageAction(QWebPage::Copy);
#endif
}


void
QLuaBrowser::doFind()
{
#if HAVE_QTWEBKIT
  if (! d->findDialog)
    d->findDialog = new FindDialog(this);
  d->findDialog->prepare();
  d->findDialog->raise();
  d->findDialog->show();
  d->findDialog->setAttribute(Qt::WA_Moved);
#endif
}


void 
QLuaBrowser::doEdit()
{
  QLuaEditor *e = 0;
  QString fileName = url().toLocalFile();
  if (QFileInfo(fileName).exists())
    e = QLuaIde::instance()->editor(fileName);
  if (e)
    return;
#if HAVE_QTWEBKIT
  QString html = d->w->page()->mainFrame()->toHtml();
  if (html.isEmpty())
    return;
  if (! (e = QLuaIde::instance()->editor()))
    return;
  e->widget()->setEditorMode("html");
  e->widget()->setPlainText(html);
#endif
}


void 
QLuaBrowser::doHome()
{
  if (d->homeUrl.isValid())
    setUrl(d->homeUrl);
}


void 
QLuaBrowser::doForward()
{
#if HAVE_QTWEBKIT
  d->w->triggerPageAction(QWebPage::Forward);
#endif
}


void 
QLuaBrowser::doBackward()
{
#if HAVE_QTWEBKIT
  d->w->triggerPageAction(QWebPage::Back);
#endif
}


void 
QLuaBrowser::doStop()
{
#if HAVE_QTWEBKIT
  d->w->triggerPageAction(QWebPage::Stop);
#endif
}


void 
QLuaBrowser::doReload()
{
#if HAVE_QTWEBKIT
  d->w->triggerPageAction(QWebPage::Reload);
#endif
}

void 
QLuaBrowser::doZoomIn()
{
#if HAVE_QTWEBKIT
  qreal z = d->w->zoomFactor() * 1.2;
  d->w->setZoomFactor(qMin(z, 4.0));
#endif
}

void 
QLuaBrowser::doZoomOut()
{
#if HAVE_QTWEBKIT
  qreal z = d->w->zoomFactor() / 1.2;
  d->w->setZoomFactor(qMax(z, 0.25));
#endif
}

void 
QLuaBrowser::doZoomReset()
{
#if HAVE_QTWEBKIT
  d->w->setZoomFactor(1.0);
#endif
}




// ========================================
// MOC


#include "qluabrowser.moc"





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
