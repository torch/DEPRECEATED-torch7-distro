/* -*- C++ -*- */

#include <QtGlobal>
#include <QAbstractListModel>
#include <QApplication>
#include <QActionGroup>
#include <QBrush>
#include <QCloseEvent>
#include <QCursor>
#include <QDateTime>
#include <QDebug>
#include <QDesktopWidget>
#include <QDir>
#include <QDialog>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontMetrics>
#include <QFontMetricsF>
#include <QFont>
#include <QFontInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QItemSelection>
#include <QItemSelectionModel>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QList>
#include <QListView>
#include <QMap>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPainter>
#include <QPalette>
#include <QPointer>
#include <QPrintDialog>
#include <QPrinter>
#include <QPushButton>
#include <QRegExp>
#include <QScrollBar>
#include <QSplitter>
#include <QSettings>
#include <QStatusBar>
#include <QString>
#include <QStringList>
#include <QTextBlock>
#include <QTextCharFormat>
#include <QTextCursor>
#include <QTimer>
#include <QToolBar>
#include <QVariant>
#include <QVariantList>
#include <QVBoxLayout>
#include <QWhatsThis>


#include "qluatextedit.h"
#include "qluamainwindow.h"
#include "qluasdimain.h"

#include "qtluaengine.h"
#include "qluaapplication.h"
#include "qluaconsole.h"
#include "qluaeditor.h"
#include "qluaide.h"

using namespace QLuaActionHelpers;


// ========================================
// QLUACONSOLEWIDGET::PRIVATE


class QLuaConsoleWidget::Private : public QObject
{
  Q_OBJECT
public:
  QLuaConsoleWidget *q;
  bool printTimings;
  bool elapsedValid;
public:
  Private(QLuaConsoleWidget *q);
protected:
  virtual bool eventFilter(QObject *o, QEvent *e);                             
public slots:
  void luaCommandEcho(QByteArray ps1, QByteArray ps2, QByteArray cmd);
  void luaConsoleOutput(QByteArray out);
  void anyoneHandlesConsoleOutput(bool &pbool);
  void acceptingCommands(bool);
  
};


QLuaConsoleWidget::Private::Private(QLuaConsoleWidget *q)
  : QObject(q),  
    q(q), 
    printTimings(false),
    elapsedValid(false)
{
}


bool 
QLuaConsoleWidget::Private::eventFilter(QObject *o, QEvent *e)
{
  if (e->type() == QEvent::KeyPress)
    {
      QKeyEvent *ke = static_cast<QKeyEvent*>(e);
      if (ke->key() == Qt::Key_Return || 
          ke->key() == Qt::Key_Enter)
        q->moveToEnd();
    }
  return false;
}


void 
QLuaConsoleWidget::Private::luaCommandEcho(QByteArray ps1, 
                                           QByteArray ps2, QByteArray cmd)
{
  // start line
  if (printTimings)
    {
      QDateTime now = QDateTime::currentDateTime();
      QString msg = tr("starting on %1"); 
      msg.prepend("\n|| ");
      msg.append("\n");
      if (q->document()->lastBlock().length() > 1)
        msg.prepend("\n");
      q->addOutput(msg.arg(now.toString()), "(console)/comment");
    }
  elapsedValid = true;
  // command
  cmd.replace("\n", "\n" + ps2).prepend(ps1);
  q->addOutput(QString::fromLocal8Bit(cmd)+"\n", "(console)/quote");
}


void 
QLuaConsoleWidget::Private::luaConsoleOutput(QByteArray out)
{
  // generic output
  q->addOutput(QString::fromLocal8Bit(out), "(console)/normal");
}


void 
QLuaConsoleWidget::Private::anyoneHandlesConsoleOutput(bool &pbool)
{
  pbool = true;
}


void 
QLuaConsoleWidget::Private::acceptingCommands(bool accepting)
{
  QLuaApplication *app = QLuaApplication::instance();
  if (accepting && app && elapsedValid)
    {
      elapsedValid = false;
      double elapsed = app->timeForLastCommand();
      QString msg = tr("Finished after %1 seconds").arg(elapsed,0,'f',3);
      emit q->statusMessage(msg);
      if (printTimings)
        {
          msg.append("\n");
          msg.prepend("|| ");
          if (q->document()->lastBlock().length() > 1)
            msg.prepend("\n");
          q->addOutput(msg, "(console)/comment");
        }
    }
}


// ========================================
// QLUACONSOLEWIDGET


QLuaConsoleWidget::~QLuaConsoleWidget()
{
  QLuaApplication *app = QLuaApplication::instance();
  disconnect(app, 0, d, 0);
  app->setupConsoleOutput();
}



QLuaConsoleWidget::QLuaConsoleWidget(QWidget *parent)
  : QLuaTextEdit(parent),
    d(new Private(this))
{
  setReadOnly(true);
  installEventFilter(d);
  QLuaApplication *app = QLuaApplication::instance();
  Q_ASSERT(app);
  connect(app, SIGNAL(luaCommandEcho(QByteArray,QByteArray,QByteArray)),
          d, SLOT(luaCommandEcho(QByteArray,QByteArray,QByteArray)) );
  connect(app, SIGNAL(luaConsoleOutput(QByteArray)),
          d, SLOT(luaConsoleOutput(QByteArray)) );
  connect(app, SIGNAL(anyoneHandlesConsoleOutput(bool&)),
          d, SLOT(anyoneHandlesConsoleOutput(bool&)) );
  connect(app, SIGNAL(acceptingCommands(bool)),
          d, SLOT(acceptingCommands(bool)) );
  // start receiving output
  app->setupConsoleOutput();
}


void 
QLuaConsoleWidget::addOutput(QString text, QTextCharFormat format)
{
  QTextBlock b = document()->lastBlock();
  QTextCursor c = textCursor();
  QScrollBar *s = verticalScrollBar();
  bool tracking = (s && s->value() >= s->maximum() - 2);
  c.setPosition(b.position());
  c.movePosition(QTextCursor::EndOfBlock);
  c.insertText(text, format);
  if (tracking)
    s->setValue(s->maximum());
}
  

bool 
QLuaConsoleWidget::printTimings() const
{
  return d->printTimings;
}


void 
QLuaConsoleWidget::setPrintTimings(bool b)
{
  d->printTimings = b;
}


void 
QLuaConsoleWidget::addOutput(QString text, QString key)
{
  addOutput(text, format(key));
}


void 
QLuaConsoleWidget::moveToEnd()
{
  QTextCursor c = textCursor();
  QTextBlock b = document()->lastBlock();
  c.setPosition(b.position());
  c.movePosition(QTextCursor::EndOfBlock);
  setTextCursor(c);
  ensureCursorVisible();
  QScrollBar *s = verticalScrollBar();
  s->setValue(s->maximum());
}





// ========================================
// QLUASDIMAIN::PRIVATE


class QLuaSdiMain::Private : public QObject
{
  Q_OBJECT
public:
  QLuaSdiMain *q;
  QSplitter *splitter;
  QLuaConsoleWidget *c;
  QLuaTextEdit *e;
  QLabel *sbLabel;
  QPrinter *printer;
  bool canUndo;
  bool canRedo;
  QPointer<QtLuaEngine> engine;
  QPointer<QDialog> findDialog;
  QPointer<QPrintDialog> printDialog;
  int historySize;
  int historyPos;
  QStringList history;
  QMap<int,QString> historyOverlay;
  QMap<int,int> historyCursorPos;
  QTimer updateTimer;
public:
  ~Private();
  Private(QLuaSdiMain *q);
protected:
  bool eventFilter(QObject *obj, QEvent *ev);
public slots:
  void newEngine();
  void blockCountChanged();
  void undoAvailable(bool);
  void redoAvailable(bool);
  void updateUndoRedo();
  void updateLuaActions();
  void processFilesToOpen();
  void luaCommandEcho(QByteArray,QByteArray,QByteArray cmd);
  bool toHistory(int n);
  void historyUp();
  void historyDown();
  void historySearch();
};
  

QLuaSdiMain::Private::~Private()
{
  delete printer;
  printer = 0;
}


QLuaSdiMain::Private::Private(QLuaSdiMain *q) 
  : QObject(q), q(q), 
    splitter(0), c(0), e(0), sbLabel(0),
    printer(0),
    canUndo(false), 
    canRedo(false),
    historySize(2000),
    historyPos(-1)
{
  splitter = new QSplitter(Qt::Vertical, q);
  c = new QLuaConsoleWidget(splitter);
  e = new QLuaTextEdit(splitter);
  splitter->setCollapsible(0, false);
  splitter->setCollapsible(1, true);
  updateTimer.setSingleShot(true);
  connect(&updateTimer, SIGNAL(timeout()), e, SLOT(update()));
}


bool 
QLuaSdiMain::Private::eventFilter(QObject *obj, QEvent *ev)
{
  if (ev->type() == QEvent::FocusIn)
    {
      if (obj == c || obj == e)
        {
          q->clearStatusMessage();
          q->updateActionsLater();
        }
    }
#ifdef Q_WS_MAC
  if (ev->type() == QEvent::ShortcutOverride && obj == e)
    {
      QKeyEvent *ke = static_cast<QKeyEvent*>(ev);
      if (ke->modifiers() == Qt::ControlModifier)
        if (ke->key() == Qt::Key_Up || 
            ke->key() == Qt::Key_Down ||
            ke->key() == Qt::Key_Return )
          return true;
    }
#endif
  return false;
}


void 
QLuaSdiMain::Private::newEngine()
{
  if (engine)
    disconnect(engine, 0, this, 0);
  engine = QLuaApplication::engine();
  if (engine)
    connect(engine, SIGNAL(stateChanged(int)),
            q, SLOT(updateStatusBar()) );
}


void
QLuaSdiMain::Private::blockCountChanged()
{
  const int MARGIN = 4;
  QRectF rf = e->blockBoundingGeometry(e->document()->begin());
  QRectF rl = e->blockBoundingGeometry(e->document()->lastBlock());
  int dh = qRound(MARGIN + MARGIN + rl.bottom() - rf.top() + 0.5);
  QList<int> sizes = splitter->sizes();
  int ch = e->viewport()->height();
  if (sizes[1] > 0)
    {
      int sh = sizes[0] + sizes[1];
      sizes[1] = qBound(0, sizes[1] + dh - ch, sh/2);
      sizes[0] = sh - sizes[1];
      splitter->setSizes(sizes);
      e->ensureCursorVisible();
      sizes = splitter->sizes();
    }
}


void 
QLuaSdiMain::Private::undoAvailable(bool b)
{
  canUndo = b;
  updateUndoRedo();
}


void 
QLuaSdiMain::Private::redoAvailable(bool b)
{
  canRedo = b;
  updateUndoRedo();
}


void 
QLuaSdiMain::Private::updateUndoRedo()
{
  bool hasFocus = e->hasFocus();
  if  (q->hasAction("ActionEditUndo"))
    q->stdAction("ActionEditUndo")->setEnabled(hasFocus && canUndo);
  if  (q->hasAction("ActionEditRedo"))
    q->stdAction("ActionEditRedo")->setEnabled(hasFocus && canRedo);
}


void 
QLuaSdiMain::Private::updateLuaActions()
{
  QLuaApplication *app = QLuaApplication::instance();
  bool accepting = app->isAcceptingCommands();
  // update lua actions
  bool okeval = false;
  if (c->hasFocus())
    okeval = c->textCursor().hasSelection();
  else if (e->hasFocus())
    okeval = true;
  if (q->hasAction("ActionLuaEval"))
    q->stdAction("ActionLuaEval")->setEnabled(accepting && okeval);
  // update status bar
  q->updateStatusBar();
  // feedback in command editor widget
  QCursor cursor((accepting) ? Qt::IBeamCursor : Qt::WaitCursor);
  e->viewport()->setCursor(cursor);
  QPalette p = e->palette();
  QPalette::ColorRole r = (accepting) ? QPalette::Base : QPalette::AlternateBase;
  p.setColor(QPalette::Base, q->palette().color(r));
  e->setPalette(p);
  if (accepting)
    e->update();
  else
    updateTimer.start(250);
}


void 
QLuaSdiMain::Private::processFilesToOpen()
{
  QLuaApplication *app = QLuaApplication::instance();
  foreach (QString s, app->filesToOpen())
    if (! s.isEmpty())
      q->openFile(s, true);
}


void 
QLuaSdiMain::Private::luaCommandEcho(QByteArray,QByteArray,QByteArray cmd)
{
  QString s = QString::fromLocal8Bit(cmd).trimmed();
  if (s.isEmpty())
    return;
  history.removeAll(s);
  history.prepend(s);
  while (history.size() > historySize)
    history.removeLast();
}


bool 
QLuaSdiMain::Private::toHistory(int n)
{
  if (n >= -1 && n < history.size())
    {
      historyOverlay[historyPos] = e->toPlainText();
      historyCursorPos[historyPos] = e->textCursor().position();
      QString s;
      if (historyOverlay.contains(n))
        s = historyOverlay[n];
      else if (n >= 0)
        s = history[n];
      int cpos = s.size();
      if (historyCursorPos.contains(n))
        cpos = historyCursorPos[n];
      if (n != historyPos)
        {
          e->setPlainText(s);
          QTextCursor c = e->textCursor();
          c.setPosition(cpos);
          e->setTextCursor(c);
          historyPos = n;
          return true;
        }
    }
  return false;
}


void 
QLuaSdiMain::Private::historyUp()
{
  toHistory(historyPos+1);
}


void 
QLuaSdiMain::Private::historyDown()
{
  toHistory(historyPos-1);
}


// ========================================
// QLUASDIMAIN HISTORY SEARCH



class QLuaSdiMain::HSModel : public QAbstractListModel
{
  Q_OBJECT
  HSWidget *w;
  Private *d;
public:
  HSModel(HSWidget *w, Private *d);
  virtual int rowCount(const QModelIndex &mi = QModelIndex()) const;
  virtual QString line(int row) const;
  virtual QVariant data(const QModelIndex &mi, int role) const;
  virtual Qt::ItemFlags flags(const QModelIndex &) const;
};


class QLuaSdiMain::HSView : public QListView
{
  Q_OBJECT
public:
  HSView(HSWidget *p, HSModel *m);
};


class QLuaSdiMain::HSWidget : public QFrame
{
  Q_OBJECT
  Private *d;
  QLineEdit *e;
  QPushButton *c;
  HSView *v;
  HSModel *m;
  QEventLoop *loop;
  int selected;
public:
  HSWidget(Private *d);
  int exec(QPoint pos);
  bool eventFilter(QObject *watched, QEvent *event);
public slots:
  void clear();
  void select(int row);
  void textChanged(const QString &text);
  void selectionChanged(const QItemSelection &s);
  void hiliteSearch();
};


QLuaSdiMain::HSModel::HSModel(HSWidget *w, Private *d)
  : QAbstractListModel(w), w(w), d(d) 
{
}


int 
QLuaSdiMain::HSModel::rowCount(const QModelIndex &) const
{
  return d->history.size();
}


QString 
QLuaSdiMain::HSModel::line(int row) const
{
  int n = d->history.size() - row - 1;
  if (d->historyOverlay.contains(n))
    return d->historyOverlay[n];
  else if (n >= 0  && n < d->history.size())
    return d->history[n];
  return QString();
}


QVariant
QLuaSdiMain::HSModel::data(const QModelIndex &mi, int role) const
{
  int row = mi.row();
  if (row >= 0 && row < rowCount())
    {
      switch(role)
        {
        case Qt::DisplayRole:
          return line(row).simplified();
        case Qt::FontRole:
          return w->font();
        case Qt::TextAlignmentRole:
          return Qt::AlignLeft;
        default:
          break;
        }
    }
  return QVariant();
}


Qt::ItemFlags
QLuaSdiMain::HSModel::flags(const QModelIndex &) const
{ 
  return Qt::ItemIsSelectable | Qt::ItemIsEnabled;
}


QLuaSdiMain::HSView::HSView(HSWidget *p, HSModel *m)
  : QListView(p)
{
  setModel(m);
  setEditTriggers(NoEditTriggers);
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
  setSelectionBehavior(SelectItems);
  setSelectionMode(SingleSelection);
  setViewMode(ListMode);
  setWrapping(false);
  setFlow(TopToBottom);
  setMovement(Static);
  setSpacing(0);
  setTextElideMode(Qt::ElideRight);
}


QLuaSdiMain::HSWidget::HSWidget(Private *d)
  : QFrame(d->q), d(d), loop(0)
{
  // layout
  QVBoxLayout *vlayout = new QVBoxLayout(this);
  QHBoxLayout *hlayout = new QHBoxLayout();
  vlayout->addLayout(hlayout);
  c = new QPushButton(this);
  c->setFlat(true);
  c->setIcon(QIcon(":images/clear.png"));
  hlayout->setSpacing(0);
  hlayout->addWidget(c);
  e = new QLineEdit(this);
  hlayout->addWidget(e);
  m = new HSModel(this, d);
  v = new HSView(this, m);
  vlayout->addWidget(v);
  setWindowFlags(Qt::Popup);
  setWindowModality(Qt::ApplicationModal);
  // focus
  c->setFocusPolicy(Qt::NoFocus);
  v->setFocusPolicy(Qt::NoFocus);
  e->installEventFilter(this);
  setFocusProxy(e);
  // appearance
  setLineWidth(2);
  setFrameStyle(Box|Plain);
  setAutoFillBackground(true);
  setBackgroundRole(QPalette::AlternateBase);
  v->setBackgroundRole(QPalette::AlternateBase);
  v->viewport()->setBackgroundRole(QPalette::AlternateBase);
  QFont font;
  const qreal fontFactor = 0.9;
  if (font.pixelSize() > 0)
    font.setPixelSize(qRound(font.pixelSize() * fontFactor + 0.5));
  else
    font.setPointSizeF(font.pointSizeF() * fontFactor);
  setFont(font);
  // connections
  connect(c, SIGNAL(clicked()), 
          this, SLOT(clear()) );
  connect(e, SIGNAL(textChanged(const QString&)),
          this, SLOT(textChanged(const QString&)) );
  connect(v->selectionModel(), 
          SIGNAL(selectionChanged(const QItemSelection&,const QItemSelection&)),
          this, SLOT(selectionChanged(const QItemSelection&)) );
}


int 
QLuaSdiMain::HSWidget::exec(QPoint pos)
{
  // initial selection
  select(d->history.size() - d->historyPos - 1);
  // position list
  QPoint tl = d->e->mapToGlobal(d->e->rect().topLeft());
  QPoint br = d->e->mapToGlobal(d->e->rect().bottomRight());
  pos = d->e->mapToGlobal(pos);
  ensurePolished();
  adjustSize();
  int rw = width() * 3 / 2;
  int rh = height();
  QRect screen = QApplication::desktop()->availableGeometry(this);
  if (tl.y() >= screen.height()-br.y())
    pos.ry() = tl.y() - rh + 4;
  if (pos.x() + rw > screen.x() + screen.width()) 
    pos.setX(screen.x() + screen.width() - rw);
  if (pos.x() < screen.x())
    pos.setX(screen.x());
  if (pos.x() + rw > screen.x() + screen.width()) 
    rw = screen.x() + screen.width() - pos.x();
  if (pos.y() + rh > screen.y() + screen.height()) 
    pos.setY(screen.y() + screen.height() - rh);
  if (pos.y() < screen.y())
    pos.setY(screen.y());
  if (pos.y() + rh > screen.y() + screen.height()) 
    rh = screen.y() + screen.height() - pos.y();
  setGeometry(pos.x(), pos.y(), rw, rh);
  // loop
  QEventLoop eventLoop;
  QPointer<QObject> guard = this;
  selected = -1;
  loop = &eventLoop;
  show();
  setFocus();
  eventLoop.exec();
  loop = 0;
  if (guard.isNull())
    return -1;
  return selected;
}


bool
QLuaSdiMain::HSWidget::eventFilter(QObject *watched, QEvent *event)
{
  if (watched == e && event->type() == QEvent::KeyPress)
    {
      switch (static_cast<QKeyEvent*>(event)->key())
        {
        case Qt::Key_Up:
        case Qt::Key_Down:
          return QCoreApplication::sendEvent(v, event);
        case Qt::Key_Escape:
          selected = -1;
        case Qt::Key_Enter:
        case Qt::Key_Return:
          if (loop)
            loop->quit();
          return true;
        default:
          break;
        }
    }
  return false;
}


void 
QLuaSdiMain::HSWidget::clear()
{
  e->setText(QString());
}


void 
QLuaSdiMain::HSWidget::select(int row)
{
  row = qBound(0, row, m->rowCount() - 1);
  QItemSelectionModel *sm = v->selectionModel();
  QModelIndex mi = m->index(row);
  sm->select(mi, QItemSelectionModel::ClearAndSelect);
  v->setCurrentIndex(mi);
  v->scrollTo(mi);
  hiliteSearch();
}


void 
QLuaSdiMain::HSWidget::selectionChanged(const QItemSelection &s)
{
  int row = -1;
  if (s.size() >= 1)
    row = s[0].top();
  selected = -1;
  if (row >= 0 && row <= d->history.size())
    if (! v->isRowHidden(row))
      selected = d->history.size() - row - 1;
  if (d->historyPos != selected)
    d->toHistory(selected);
  hiliteSearch();
}


void 
QLuaSdiMain::HSWidget::textChanged(const QString &text)
{
  int f = -1;
  int s = -1;
  int c = d->history.size() - selected - 1;
  for (int i=0; i <= d->history.size(); i++)
    {
      QString str = m->line(i);
      bool ok = str.contains(text);
      v->setRowHidden(i, !ok);
      if (ok)
        f = i;
      if (ok && i <= c)
        s = i;
    }
  if (s < 0 && f >= 0)
    s = f;
  if (s >= 0) 
    select(s);
}


void
QLuaSdiMain::HSWidget::hiliteSearch()
{
  QList<QTextEdit::ExtraSelection> sel;
  QString search = e->text();
  if (search.size() > 0)
    {
      int pos = 0;
      QString line = d->e->toPlainText();
      while ((pos = line.indexOf(search, pos)) >= 0)
        {
          QTextEdit::ExtraSelection extra;
          extra.cursor = d->e->textCursor();
          extra.cursor.setPosition(pos);
          pos += search.size();
          extra.cursor.setPosition(pos, QTextCursor::KeepAnchor);
          extra.format = d->e->format("(matcher)/ok");
          sel += extra;
        }
    }
  d->e->setExtraSelections(sel);
}


void 
QLuaSdiMain::Private::historySearch()
{
  historyOverlay.clear();
  historyCursorPos.clear();
  int pos = historyPos;
  HSWidget *hsw = new HSWidget(this);
  int selected = hsw->exec(e->cursorRect().bottomRight());
  delete hsw;
  q->activateWindow();
  if (selected >= 0 && selected < history.size())
    pos = selected;
  toHistory(pos);
}






// ========================================
// QLUASDIMAIN


QLuaSdiMain::QLuaSdiMain(QWidget *parent)
  : QLuaMainWindow("qLuaSdiMain", parent),
    d(new Private(this))
{
  // layout
  setCentralWidget(d->splitter);
  setFocusProxy(d->e);
  d->e->setEditorMode("lua");
  d->e->setFocus();
  menuBar();
  toolBar();
  statusBar();
  // settings
  loadSettings();
  // title
  QString appName = QCoreApplication::applicationName();
  setWindowTitle(tr("%1 Console").arg(appName));
  // engine and app signals
  QLuaApplication *app = QLuaApplication::instance();
  connect(app, SIGNAL(acceptingCommands(bool)), 
          d, SLOT(updateLuaActions()) );
  connect(app, SIGNAL(fileOpenEvent()),
          d, SLOT(processFilesToOpen()) );
  connect(app, SIGNAL(luaCommandEcho(QByteArray,QByteArray,QByteArray)),
          d, SLOT(luaCommandEcho(QByteArray,QByteArray,QByteArray)) );
  connect(app, SIGNAL(newEngine(QtLuaEngine*)),
          d, SLOT(newEngine()) );
  d->newEngine();
  // editor signals
  d->e->installEventFilter(d);
  connect(d->e->document(), SIGNAL(blockCountChanged(int)),
          d, SLOT(blockCountChanged()) );
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
          this, SLOT(clearStatusMessage()) );
  // console signals
  d->c->installEventFilter(d);
  connect(d->c, SIGNAL(settingsChanged()),
          this, SLOT(updateActionsLater()) );
  connect(d->c, SIGNAL(selectionChanged()),
          this, SLOT(updateActionsLater()) );
  connect(d->c, SIGNAL(statusMessage(const QString&)),
          this, SLOT(showStatusMessage(const QString&)) );
  // update
  updateActions();
  QTimer::singleShot(0, d, SLOT(blockCountChanged()));
  QTimer::singleShot(10, d, SLOT(processFilesToOpen()));
}


void
QLuaSdiMain::loadSettings()
{
  QSettings s;
  QLuaTextEdit *e = d->e;
  QLuaConsoleWidget *c = d->c;
  // font settings
  QFont font = QApplication::font();
  if  (s.contains("console/font"))
    font = qvariant_cast<QFont>(s.value("console/font"));
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
  QFontMetrics metrics(font);
  e->setMinimumSize(QSize(metrics.width("MM"), metrics.lineSpacing()*3));
  e->setFont(font);
  c->setFont(font);
  // sizes
  int cLines = s.value("console/consoleLines", 10000).toInt();
  c->setMaximumBlockCount(qMax(100,cLines));
  QSize cSize = s.value("console/consoleSize", QSize(80,25)).toSize();
  c->setSizeInChars(cSize);
  QSize eSize = s.value("console/editorSize", QSize(80,2)).toSize();
  e->setSizeInChars(eSize);
  // tabs
  int tabSize = -1;
  if (s.contains("console/tabSize"))
    tabSize = s.value("console/tabSize").toInt();
  if (tabSize<2 || tabSize>16)
    tabSize = 8;
  e->setTabSize(tabSize);
  c->setTabSize(tabSize);
  // other 
  e->setTabExpand(s.value("console/tabExpand", true).toBool());
  e->setAutoIndent(s.value("console/autoIndent", true).toBool());
  e->setAutoComplete(s.value("console/autoComplete", true).toBool());
  e->setAutoMatch(s.value("console/autoMatch", true).toBool());
  e->setAutoHighlight(s.value("console/autoHighlight", true).toBool());
  c->setLineWrapMode(s.value("console/lineWrap",true).toBool() ?
                     QLuaTextEdit::WidgetWidth : QLuaTextEdit::NoWrap);
  // splitter
  d->splitter->restoreState(s.value("console/splitter").toByteArray());
  // history
  d->historySize = qMax(0, s.value("console/historySize", 1000).toInt());
  QSettings hs( QDir::homePath() + "/.qluahistory", QSettings::IniFormat);
  d->history = hs.value("history").toStringList();
  // Inherit
  QLuaMainWindow::loadSettings();
}


void
QLuaSdiMain::saveSettings()
{
  QLuaMainWindow::saveSettings();
  QSettings s;
  QLuaTextEdit *e = d->e;
  // misc
  s.setValue("console/autoComplete", e->autoComplete());
  s.setValue("console/autoIndent", e->autoIndent());
  s.setValue("console/autoMatch", e->autoMatch());
  s.setValue("console/autoHighlight", e->autoHighlight());
  s.setValue("console/lineWrap", d->c->lineWrapMode() != QLuaTextEdit::NoWrap);
  s.setValue("console/splitter", d->splitter->saveState());
  // history
  s.setValue("console/historySize", d->historySize);
  QSettings hs( QDir::homePath() + "/.qluahistory", QSettings::IniFormat);
  hs.setValue("history", d->history);
}

int 
QLuaSdiMain::historySize() const
{
  return d->historySize;
}

int 
QLuaSdiMain::consoleLines() const
{
  return d->c->maximumBlockCount();
}

void 
QLuaSdiMain::setHistorySize(int n)
{
  d->historySize = qMax(n, 0);
}

void 
QLuaSdiMain::setConsoleLines(int n)
{
  d->c->setMaximumBlockCount(qMax(n,256));
}


QLuaConsoleWidget *
QLuaSdiMain::consoleWidget()
{
  return d->c;
}


QLuaTextEdit *
QLuaSdiMain::editorWidget()
{
  return d->e;
}



QAction *
QLuaSdiMain::createAction(QByteArray name)
{
  if (name == "MenuFile")
    {
      QMenu *menu = newMenu(tr("&File", "file|"));
      menu->addAction(stdAction("ActionFileNew"));
      menu->addAction(stdAction("ActionFileOpen"));
      menu->addAction(stdAction("MenuOpenRecent"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionFileSaveAs"));
      menu->addAction(stdAction("ActionFilePrint"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionPreferences"));
      menu->addSeparator();
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
      menu->addAction(stdAction("ActionEditFind"));
      menu->addAction(stdAction("ActionConsoleClear"));
      return menu->menuAction();
    } 
  else if (name == "MenuTools")
    {
      QMenu  *menu = newMenu(tr("&Tools", "tools|"));
      menu->addAction(stdAction("ActionLineWrap"));
      menu->addAction(stdAction("ActionModeComplete"));
      menu->addAction(stdAction("ActionModeAutoIndent"));
      menu->addAction(stdAction("ActionModeAutoHighlight"));
      menu->addAction(stdAction("ActionModeAutoMatch"));
      return menu->menuAction();
    } 
  else if (name == "MenuLua")
    {
      QMenu *menu = newMenu(tr("&Lua", "lua|"));
      menu->addAction(stdAction("ActionHistoryUp"));
      menu->addAction(stdAction("ActionHistoryDown"));
      menu->addAction(stdAction("ActionHistorySearch"));
      menu->addSeparator();
      menu->addAction(stdAction("ActionLuaEval"));
      menu->addAction(stdAction("ActionLuaPause"));
      menu->addAction(stdAction("ActionLuaStop"));
      return menu->menuAction();
    } 
  // items
  else if (name == "ActionFileSaveAs")
    {
      return QLuaMainWindow::createAction(name)
        << QIcon(":/images/filesave.png")
        << QKeySequence(QKeySequence::Save)
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
  else if (name == "ActionEditFind")
    {
      return QLuaMainWindow::createAction(name)
        << Connection(this, SLOT(doFind()));
    }
  else if (name == "ActionConsoleClear")
    {
      return newAction(tr("&Clear Console", "tools|clear"))
        << Connection(this, SLOT(doClear()))
        << QIcon(":/images/clear.png")
        << tr("Clear all text in console window.");
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
  else if (name == "ActionHistoryUp")
    {
      return newAction(tr("&Up History","history|up"))
        << Connection(d, SLOT(historyUp()))
        << QKeySequence(tr("Ctrl+Up","history|up"))
        << QIcon(":/images/up.png")
        << tr("Previous item in command history.");
    }
    else if (name == "ActionHistoryDown")
    {
      return newAction(tr("&Down History","history|down"))
        << Connection(d, SLOT(historyDown()))
        << QKeySequence(tr("Ctrl+Down","history|down"))
        << QIcon(":/images/down.png")
        << tr("Next item in command history.");
    }
  else if (name == "ActionHistorySearch")
    {
      return newAction(tr("&Search History","history|search"))
        << Connection(d, SLOT(historySearch()))
        << QKeySequence(tr("Ctrl+R","history|search"))
        << QIcon(":/images/history.png")
        << tr("Search command history.");
    }
  else if (name == "ActionLuaEval")
    {
      return newAction(tr("&Eval Lua Expression","lua|eval"))
        << QKeySequence(tr("Ctrl+E","lua|eval"))
        << QKeySequence(tr("Ctrl+Return", "lua|load"))
        << QKeySequence(tr("Ctrl+Enter","lua|load"))
        << QIcon(":/images/playerplay.png")
        << Connection(this, SLOT(doEval()))
        << tr("Evaluate the Lua expression.");
    }
  // default
  return QLuaMainWindow::createAction(name);
}


QToolBar*
QLuaSdiMain::createToolBar()
{
  QToolBar *toolBar = new QToolBar(this);
  toolBar->addAction(stdAction("ActionFileNew"));
  toolBar->addAction(stdAction("ActionFileOpen"));
  toolBar->addAction(stdAction("ActionFileSaveAs"));
  toolBar->addAction(stdAction("ActionFilePrint"));
  toolBar->addSeparator();
#ifndef Q_WS_MAC
  toolBar->addAction(stdAction("ActionEditUndo"));
  toolBar->addAction(stdAction("ActionEditRedo"));
#endif
  toolBar->addAction(stdAction("ActionModeBalance"));
  toolBar->addAction(stdAction("ActionEditFind"));
  toolBar->addAction(stdAction("ActionConsoleClear"));
  toolBar->addSeparator();
  toolBar->addAction(stdAction("ActionHistorySearch"));
  toolBar->addAction(stdAction("ActionLuaEval"));
  toolBar->addAction(stdAction("ActionLuaPause"));
  toolBar->addAction(stdAction("ActionLuaStop"));
  if (! hasAction("ActionWhatsThis"))
    return toolBar;
  toolBar->addSeparator();
  toolBar->addAction(stdAction("ActionWhatsThis"));
  return toolBar;
}


QMenuBar*
QLuaSdiMain::createMenuBar()
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


QStatusBar*
QLuaSdiMain::createStatusBar()
{
  QStatusBar *statusbar = new QStatusBar(this);
  QFont font = QApplication::font();
  QFontMetrics metric(font);
  d->sbLabel = new QLabel();
  d->sbLabel->setFont(font);
  d->sbLabel->setAlignment(Qt::AlignCenter);
  d->sbLabel->setMinimumWidth(metric.width(" XXXX "));
  statusbar->addPermanentWidget(d->sbLabel);
  return statusbar;
}


bool 
QLuaSdiMain::canClose()
{
  QLuaIde *ide = QLuaIde::instance();
  if (ide && !ide->quit(this))
    return false;
  return true;
}


bool
QLuaSdiMain::newDocument()
{
  QLuaEditor *n = QLuaIde::instance()->editor();
  QLuaTextEditMode *mode = d->e->editorMode();
  n->widget()->setEditorMode(mode ? mode->factory() : 0);
  n->updateActionsLater();
  return true;
}


void
QLuaSdiMain::doSaveAs()
{
  QString msg = tr("Save Console Data");
  QString dir = QDir::currentPath();
  QString f = QLuaIde::fileDialogFilters();
  QString s = QLuaIde::allFilesFilter();
  QFileDialog::Options o = QFileDialog::DontUseNativeDialog;
  QString fname = QFileDialog::getSaveFileName(window(), msg, dir, f, &s, o);
  if (fname.isEmpty())
    return;
  QFile file(fname);
  if (d->c->writeFile(file))
    return;
  QString an = QCoreApplication::applicationName();
  QString ms = tr("<html>Cannot save file \"%1\".&nbsp;&nbsp;"
                  "<br>%2.</html>")
    .arg(QFileInfo(file).fileName())
    .arg(file.errorString());
  QMessageBox::critical(this, tr("%1 Editor -- Error").arg(an), ms);
}


void
QLuaSdiMain::doPrint()
{
  QPrinter *printer = loadPageSetup();
  if (! d->printDialog)
    d->printDialog = new QPrintDialog(printer, this);
  QPrintDialog::PrintDialogOptions options = d->printDialog->enabledOptions();
  options &= ~QPrintDialog::PrintSelection;
  if (d->c->textCursor().hasSelection())
    options |= QPrintDialog::PrintSelection;
  d->printDialog->setEnabledOptions(options);
  if (d->printDialog->exec() == QDialog::Accepted)
    {
      d->c->print(printer);
      savePageSetup();
    }
}


void
QLuaSdiMain::doSelectAll()
{
  if (d->e->hasFocus())
    d->e->selectAll();
  else
    d->c->selectAll();    
  updateActionsLater();
}


void
QLuaSdiMain::doUndo()
{
  if (d->e->hasFocus())
    d->e->undo();
  updateActionsLater();
}


void
QLuaSdiMain::doRedo()
{
  if (d->e->hasFocus())
    d->e->redo();
  updateActionsLater();
}


void
QLuaSdiMain::doCut()
{
  if (d->e->hasFocus())
    d->e->cut();
  updateActionsLater();
}


void
QLuaSdiMain::doCopy()
{
  if (d->e->hasFocus())
    d->e->copy();
  else
    d->c->copy();
  updateActionsLater();
}


void
QLuaSdiMain::doPaste()
{
  if (d->e->hasFocus())
    d->e->paste();
  updateActionsLater();
}


void
QLuaSdiMain::doFind()
{
  QDialog *dialog = d->findDialog;
  if (! dialog)
    d->findDialog = dialog = d->c->makeFindDialog();
  d->c->prepareDialog(dialog);
  dialog->raise();
  dialog->show();
  dialog->setAttribute(Qt::WA_Moved);
}


void
QLuaSdiMain::doLineWrap(bool b)
{
  if (b)
    d->c->setLineWrapMode(QLuaTextEdit::WidgetWidth);
  else
    d->c->setLineWrapMode(QLuaTextEdit::NoWrap);
  updateActionsLater();
}


void
QLuaSdiMain::doHighlight(bool b)
{
  d->e->setAutoHighlight(b);
  updateActionsLater();
}


void 
QLuaSdiMain::doCompletion(bool b)
{
  d->e->setAutoComplete(b);
  updateActionsLater();
}


void
QLuaSdiMain::doAutoIndent(bool b)
{
  d->e->setAutoIndent(b);
  updateActionsLater();
}


void
QLuaSdiMain::doAutoMatch(bool b)
{
  d->e->setAutoMatch(b);
  updateActionsLater();
}


void
QLuaSdiMain::doBalance()
{
  if (d->e->hasFocus())
    {
      QLuaTextEditMode *mode = d->e->editorMode();
      if (mode && mode->supportsBalance() && mode->doBalance())
        return;
      showStatusMessage(tr("Cannot find enclosing expression."), 5000);
      QLuaApplication::beep();
    }
}


void
QLuaSdiMain::doClear()
{
  d->c->selectAll();
  d->c->insertPlainText(QString());
}


void
QLuaSdiMain::doEval()
{
  QTextCursor cursor;
  if (d->c->hasFocus()) 
    {
      cursor = d->c->textCursor();
    }
  else if (d->e->hasFocus())
    {
      d->historyPos = -1;
      d->historyOverlay.clear();
      d->historyCursorPos.clear();
      d->c->moveToEnd();
      cursor = d->e->textCursor();
      if (! cursor.hasSelection())
        {
          d->e->selectAll();
          cursor = d->e->textCursor();
        }
    }
  if (cursor.hasSelection())
    {
      QString s = cursor.selectedText().trimmed();
      s = s.replace(QChar(0x2029),QChar('\n'));
      s = s.replace(QRegExp("^\\s*=\\s*"), "return ");
      if (QLuaIde::instance()->luaExecute(s.toLocal8Bit()))
        return;
    }
  showStatusMessage(tr("Nothing to evaluate here."), 5000);
  QLuaApplication::beep();
}


void
QLuaSdiMain::updateStatusBar()
{
  QLuaApplication *app = QLuaApplication::instance();
  QtLuaEngine *engine = d->engine;
  if (d->sbLabel)
    {
      QStringList modes;
      if (app->isAcceptingCommands() && engine && engine->isReady())
        modes += tr("Ready", "status bar indicator");
      else if (engine && engine->isPaused())
        modes += tr("Paused", "status bar indicator");
      else
        modes += tr("Running", "status bar indicator");
      // overwrite
      if (d->e->overwriteMode())
        modes += tr("Ovrw", "status bar indicator");
      QString s = " " + modes.join(" ") + " ";
      if (s != d->sbLabel->text())
        d->sbLabel->setText(s);
    }
}


void
QLuaSdiMain::updateActions()
{
  QLuaMainWindow::updateActions();
  // undo redo
  d->updateUndoRedo();
  // cut copy paste
  QLuaTextEdit *a = qobject_cast<QLuaTextEdit*>(focusWidget());
  bool readOnly = (a) ? a->isReadOnly() : true;
  bool canPaste = (a) ? a->canPaste() : false;
  bool canCopy = (a) ? a->textCursor().hasSelection() : false;
  if (hasAction("ActionEditPaste"))
    stdAction("ActionEditPaste")->setEnabled(canPaste && !readOnly);
  if (hasAction("ActionEditCut"))
    stdAction("ActionEditCut")->setEnabled(canCopy && !readOnly);
  if (hasAction("ActionEditCopy"))
    stdAction("ActionEditCopy")->setEnabled(canCopy);
  // misc
  bool eHasFocus = d->e->hasFocus();
  bool cHasFocus = d->c->hasFocus();
  if (hasAction("ActionConsoleClear"))
    stdAction("ActionConsoleClear")->setEnabled(cHasFocus);
  if (hasAction("ActionEditFind"))
    stdAction("ActionEditFind")->setEnabled(cHasFocus);
  if (hasAction("ActionModeBalance"))
    stdAction("ActionModeBalance")->setEnabled(eHasFocus);
  // tools
  bool wrap = (d->c->lineWrapMode() != QLuaTextEdit::NoWrap);
  if (hasAction("ActionLineWrap"))
    stdAction("ActionLineWrap")->setChecked(wrap);
  if (hasAction("ActionModeAutoHighlight"))
    stdAction("ActionModeAutoHighlight")->setChecked(d->e->autoHighlight());
  if (hasAction("ActionModeAutoMatch"))
    stdAction("ActionModeAutoMatch")->setChecked(d->e->autoMatch());
  if (hasAction("ActionModeAutoIndent"))
    stdAction("ActionModeAutoIndent")->setChecked(d->e->autoIndent());
  if (hasAction("ActionModeComplete"))
    stdAction("ActionModeComplete")->setChecked(d->e->autoComplete());
  // history
  int hp = d->historyPos;
  int hs = d->history.size();
  if (hasAction("ActionHistoryUp"))
    stdAction("ActionHistoryUp")->setEnabled(eHasFocus && (hp+1 < hs) );
  if (hasAction("ActionHistoryDown"))
    stdAction("ActionHistoryDown")->setEnabled(eHasFocus && (hp >= 0));
  if (hasAction("ActionHistorySearch"))
    stdAction("ActionHistorySearch")->setEnabled(eHasFocus && (hs > 0));
  // lua actions
  d->updateLuaActions();
}






// ========================================
// MOC


#include "qluasdimain.moc"





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
