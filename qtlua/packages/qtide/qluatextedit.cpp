/* -*- C++ -*- */

#include <QtGlobal>
#include <QAbstractTextDocumentLayout>
#include <QApplication>
#include <QActionGroup>
#include <QCloseEvent>
#include <QDebug>
#include <QDialog>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontMetrics>
#include <QFont>
#include <QFontInfo>
#include <QKeyEvent>
#include <QLineEdit>
#include <QList>
#include <QMap>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPainter>
#include <QPaintEvent>
#include <QPointer>
#include <QPrinter>
#include <QPushButton>
#include <QRegExp>
#include <QSettings>
#include <QShortcut>
#include <QString>
#include <QStringList>
#include <QSyntaxHighlighter>
#include <QTemporaryFile>
#include <QTextBlock>
#include <QTextBlockUserData>
#include <QTextEdit>
#include <QTextCharFormat>
#include <QTextCursor>
#include <QTextFrameFormat>
#include <QTextLayout>
#include <QTextOption>
#include <QTimer>
#include <QToolBar>
#include <QVariant>
#include <QVBoxLayout>
#include <QWhatsThis>


#include "qluatextedit.h"




// ========================================
// GOTO DIALOG


#include "ui_qluagotodialog.h"

class QLuaTextEdit::GotoDialog : public QDialog
{
  Q_OBJECT
protected:
  Ui_QLuaGotoDialog ui;
  QLuaTextEdit *editor;
public:
  GotoDialog(QLuaTextEdit *editor);
  void accept();
  void prepare();
};


QLuaTextEdit::GotoDialog::GotoDialog(QLuaTextEdit *editor)
  : QDialog(editor), editor(editor)
{
  ui.setupUi(this);
  ui.buttonBox->button(QDialogButtonBox::Ok)->setDefault(true);
}


void
QLuaTextEdit::GotoDialog::prepare()
{
  QTextDocument *d = editor->document();
  QTextCursor c = editor->textCursor();
  ui.spinBox->setMinimum(1);
  ui.spinBox->setMaximum(d->blockCount());
  ui.spinBox->setValue(c.blockNumber()+1);
  ui.spinBox->selectAll();
}


void
QLuaTextEdit::GotoDialog::accept()
{
  QDialog::accept();
  int lineno = ui.spinBox->value();
  QTextDocument *d = editor->document();
  QTextCursor c = editor->textCursor();
  QTextBlock b = d->findBlockByNumber(lineno-1);
  if (b.isValid())
    {
      c.setPosition(b.position());
      editor->setTextCursor(c);
      editor->ensureCursorVisible();
      return;
    }
  QApplication::beep();
}




// ========================================
// FIND DIALOG


#include "ui_qluafinddialog.h"


class QLuaTextEdit::FindDialog : public QDialog
{
  Q_OBJECT
protected:
  Ui_QLuaFindDialog ui;
  QLuaTextEdit *editor;
  QTextDocument *document;
  QPointer<QShortcut> findNextSCut;
  QPointer<QShortcut> findPrevSCut;
public:
  FindDialog(QLuaTextEdit *editor);
  ~FindDialog();
  void prepare();
protected slots:
  void update();
  bool find(bool);
  void findNext()     { find(false); }
  void findPrevious() { find(true); }
  void next();
};


QLuaTextEdit::FindDialog::FindDialog(QLuaTextEdit *editor)
  : QDialog(editor), 
    editor(editor), 
    document(editor->document())
{
  // ui
  ui.setupUi(this);
  connect(ui.findButton,SIGNAL(clicked()),this, SLOT(next()));
  connect(ui.findEdit,SIGNAL(textChanged(QString)), this, SLOT(update())); 
  connect(ui.findEdit,SIGNAL(returnPressed()), this, SLOT(next())); 
  new QShortcut(QKeySequence::FindNext, this, SLOT(findNext()));
  new QShortcut(QKeySequence::FindPrevious, this, SLOT(findPrevious()));
  findNextSCut = new QShortcut(QKeySequence::FindNext, editor);
  findPrevSCut = new QShortcut(QKeySequence::FindPrevious, editor);
  connect(findNextSCut, SIGNAL(activated()), this, SLOT(findNext()));
  connect(findPrevSCut, SIGNAL(activated()), this, SLOT(findPrevious()));
  update();
  // settings
  QSettings s;
  bool c = s.value("editor/find/caseSensitive",false).toBool();
  bool w = s.value("editor/find/wholeWords",true).toBool();
  ui.caseSensitiveBox->setChecked(c);
  ui.wholeWordsBox->setChecked(w);
}


QLuaTextEdit::FindDialog::~FindDialog()
{
  delete findNextSCut;
  delete findPrevSCut;
  QSettings s;
  s.setValue("editor/find/caseSensitive", ui.caseSensitiveBox->isChecked());
  s.setValue("editor/find/wholeWords", ui.wholeWordsBox->isChecked());
}


void
QLuaTextEdit::FindDialog::prepare()
{
  QTextCursor cursor = editor->textCursor();
  if (cursor.hasSelection())
    {
      QString s = cursor.selectedText();
      if (s.contains(QChar(0x2029)))
        s.truncate(s.indexOf(QChar(0x2029)));
      if (! s.isEmpty()) 
        ui.findEdit->setText(s);
      ui.findEdit->selectAll();
      ui.findEdit->setFocus();
    }
  update();
}


void
QLuaTextEdit::FindDialog::update()
{
  ui.findButton->setEnabled(! ui.findEdit->text().isEmpty());
}


bool
QLuaTextEdit::FindDialog::find(bool backward)
{
  if (ui.findEdit->text().isEmpty())
    return false;
  QTextDocument::FindFlags flags = 0;
  if (backward)
    flags |= QTextDocument::FindBackward;
  if (ui.caseSensitiveBox->isChecked())
    flags |= QTextDocument::FindCaseSensitively;
  if (ui.wholeWordsBox->isChecked())
    flags |= QTextDocument::FindWholeWords;
  QTextCursor cursor = editor->textCursor();
  cursor = document->find(ui.findEdit->text(), cursor, flags);
  if (cursor.isNull())
    cursor = document->find(ui.findEdit->text(), cursor, flags);
  if (cursor.isNull())
    return false;
  editor->setTextCursor(cursor);
  editor->ensureCursorVisible();
  update();
  return true;
}


void
QLuaTextEdit::FindDialog::next()
{
  if (! find(ui.searchBackwardsBox->isChecked()))
    QMessageBox::warning(this, tr("Find Warning"), 
                         tr("Search text not found."));    
}




// ========================================
// REPLACE DIALOG


#include "ui_qluareplacedialog.h"


class QLuaTextEdit::ReplaceDialog : public QDialog
{
  Q_OBJECT
protected:
  Ui_QLuaReplaceDialog ui;
  QLuaTextEdit *editor;
  QTextDocument *document;
  bool firstTime;
public:
  ReplaceDialog(QLuaTextEdit *editor);
  ~ReplaceDialog();
  void prepare();
protected slots:
  void update();
  bool find(bool);
  void findNext()     { find(false); }
  void findPrevious() { find(true); }
  void next();
  void replace();
  void replaceAll();
  
};


QLuaTextEdit::ReplaceDialog::ReplaceDialog(QLuaTextEdit *editor)
  : QDialog(editor), 
    editor(editor), 
    document(editor->document())
{
  ui.setupUi(this);
  update();
  connect(ui.findButton,SIGNAL(clicked()),this, SLOT(next()));
  connect(ui.replaceButton,SIGNAL(clicked()),this, SLOT(replace()));
  connect(ui.replaceAllButton,SIGNAL(clicked()),this, SLOT(replaceAll()));
  connect(ui.findEdit,SIGNAL(textChanged(QString)), this, SLOT(update())); 
  connect(ui.findEdit,SIGNAL(returnPressed()), this, SLOT(next())); 
  new QShortcut(QKeySequence::FindNext, this, SLOT(findNext()));
  new QShortcut(QKeySequence::FindPrevious, this, SLOT(findPrevious()));
  QSettings s;
  bool c = s.value("editor/find/caseSensitive",false).toBool();
  bool w = s.value("editor/find/wholeWords",true).toBool();
  ui.caseSensitiveBox->setChecked(c);
  ui.wholeWordsBox->setChecked(w);
}


QLuaTextEdit::ReplaceDialog::~ReplaceDialog()
{
  QSettings s;
  s.setValue("editor/find/caseSensitive", ui.caseSensitiveBox->isChecked());
  s.setValue("editor/find/wholeWords", ui.wholeWordsBox->isChecked());
}


void
QLuaTextEdit::ReplaceDialog::prepare()
{
  QTextCursor cursor = editor->textCursor();
  if (cursor.hasSelection())
    {
      QString s = cursor.selectedText();
      if (s.contains(QChar(0x2029)))
        s.truncate(s.indexOf(QChar(0x2029)));
      if (! s.isEmpty())
        ui.findEdit->setText(s);
      ui.findEdit->selectAll();
      ui.findEdit->setFocus();
    }
  update();
}


void
QLuaTextEdit::ReplaceDialog::update()
{
  ui.findButton->setEnabled(!ui.findEdit->text().isEmpty());
  bool match = (ui.findEdit->text() == editor->textCursor().selectedText());
  ui.replaceButton->setEnabled(match);
  ui.replaceAllButton->setEnabled(match);
}


bool
QLuaTextEdit::ReplaceDialog::find(bool backwards)
{
  if (ui.findEdit->text().isEmpty())
    return false;
  QTextDocument::FindFlags flags = 0;
  if (backwards)
    flags |= QTextDocument::FindBackward;
  if (ui.caseSensitiveBox->isChecked())
    flags |= QTextDocument::FindCaseSensitively;
  if (ui.wholeWordsBox->isChecked())
    flags |= QTextDocument::FindWholeWords;
  QTextCursor cursor = editor->textCursor();
  cursor = document->find(ui.findEdit->text(), cursor, flags);
  if (cursor.isNull())
    cursor = document->find(ui.findEdit->text(), cursor, flags);
  if (cursor.isNull())
    return false;
  editor->setTextCursor(cursor);
  editor->ensureCursorVisible();
  update();
  return true;
}


void
QLuaTextEdit::ReplaceDialog::next()
{
  if (! find(ui.searchBackwardsBox->isChecked()))
    QMessageBox::warning(this, tr("Replace Warning"), 
                         tr("Search text not found."));    
}

void
QLuaTextEdit::ReplaceDialog::replace()
{
  QTextCursor cursor = editor->textCursor();
  cursor.insertText(ui.replaceEdit->text());
  next();
}


void
QLuaTextEdit::ReplaceDialog::replaceAll()
{
  QTextDocument::FindFlags flags = 0;
  if (ui.caseSensitiveBox->isChecked())
    flags |= QTextDocument::FindCaseSensitively;
  if (ui.wholeWordsBox->isChecked())
    flags |= QTextDocument::FindWholeWords;
  QTextCursor cursor(document);
  QString ftext = ui.findEdit->text();
  QString rtext = ui.replaceEdit->text();
  int n = 0;
  while(! (cursor = document->find(ftext, cursor, flags)).isNull()) {
    cursor.insertText(rtext);
    n += 1;
  }
  update();
  QMessageBox::warning(this, tr("Replace All"), 
                       tr("Replaced %n occurrence(s).", 0, n));
}





// ========================================
// QLUATEXTEDITMODE



QLuaTextEditMode::QLuaTextEditMode(QLuaTextEditModeFactory *f, QLuaTextEdit *e)
  : e(e), f(f)
{
}


QLuaTextEditModeFactory *QLuaTextEditModeFactory::first = 0;
QLuaTextEditModeFactory *QLuaTextEditModeFactory::last = 0;


QLuaTextEditModeFactory::~QLuaTextEditModeFactory()
{
  if (prev) { prev->next = next; } else { first = next; }
  if (next) { next->prev = prev; } else { last = prev; }
}


QLuaTextEditModeFactory::QLuaTextEditModeFactory(const char *name, 
                                                 const char *suffixes)
  : name_(name), suffixes_(suffixes), next(first), prev(0)
{
  // find sorted position
  while (next && strcmp(next->name_, name_) < 0)
    { prev = next; next = next->next; }
  // insert
  if (prev) { prev->next = this; } else { first = this; }
  if (next) { next->prev = this; } else { last = this; }
}


QString 
QLuaTextEditModeFactory::name()
{
  return QLuaTextEditMode::tr(name_);
}


QString 
QLuaTextEditModeFactory::filter()
{
  QString patterns = "*." + suffixes().join(" *.");
  return QLuaTextEditMode::tr("%1 Files (%2)").arg(name()).arg(patterns);
}


QStringList 
QLuaTextEditModeFactory::suffixes()
{
  return QString(suffixes_).split(';');
}


QList<QLuaTextEditModeFactory*> 
QLuaTextEditModeFactory::factories()
{
  QList<QLuaTextEditModeFactory*> factories;
  for (QLuaTextEditModeFactory *f = first; f; f=f->next)
    factories += f;
  return factories;
}



// ========================================
// LINENUMBERS


class QLuaTextEdit::LineNumbers : public QWidget
{
  Q_OBJECT
  QLuaTextEdit *e;
public:
  LineNumbers(QLuaTextEdit *e, QWidget *p);
  virtual void paintEvent(QPaintEvent *event);
  virtual QSize sizeHint() const;
public slots:
  void updateRequest(const QRect &rect, int dy);
};


QLuaTextEdit::LineNumbers::LineNumbers(QLuaTextEdit *e, QWidget *p)
  : QWidget(p), e(e)
{
  connect(e, SIGNAL(updateRequest(const QRect&, int)),
          this, SLOT(updateRequest(const QRect&, int)) );
}

QSize
QLuaTextEdit::LineNumbers::sizeHint() const
{
  QString s;
  s.setNum(qMax(1000, e->blockCount()+1));
  s = QString(s.size()+2,QChar('0'));
  QFont font = e->font();
  font.setWeight(QFont::Bold);
  QFontMetrics metrics(font);
  return QSize(metrics.width(s), 0);
}


void
QLuaTextEdit::LineNumbers::updateRequest(const QRect &rect, int dy)
{
  if (isHidden())
    return;
  if (dy != 0)
    scroll(0, dy);
  else
    {
      QRect r = rect;
      r.setLeft(0);
      r.setWidth(width());
      QPointF offset = e->contentOffset();
      for (QTextBlock block = e->firstVisibleBlock(); 
           block.isValid(); block=block.next() )
        {
          if (! block.isVisible())
            continue;
          QRectF rf = e->blockBoundingGeometry(block).translated(offset);
          if (rf.bottom() < rect.top())
            continue;
          if (rf.top() > rect.bottom())
            break;
          r.setTop(qMin(r.top(), (int)rf.top()));
          r.setBottom(qMax(r.bottom(), (int)rf.bottom()));
        }
      update(r);
    }
}


void
QLuaTextEdit::LineNumbers::paintEvent(QPaintEvent *event)
{
  QRect er = event->rect();
  QPointF offset = e->contentOffset();
  QPainter painter(this);
  QFont font = e->font();
  QTextBlock sblock = e->textCursor().block();
  font.setWeight(QFont::Bold);
  painter.setFont(font);
  for (QTextBlock block = e->firstVisibleBlock(); 
       block.isValid(); block=block.next() )
    {
      int n = block.blockNumber();
      if (! block.isVisible())
        continue;
      QRectF r = e->blockBoundingGeometry(block).translated(offset);
      if (r.bottom() < er.top())
        continue;
      if (r.top() > er.bottom())
        break;
      r.setLeft(0);
      r.setWidth(width());
      QString s = QString("%1 ").arg(n+1);
      painter.setPen(block == sblock ? Qt::black : Qt::darkGray);
      painter.drawText(r, Qt::AlignRight|Qt::AlignTop|Qt::TextSingleLine, s);
    }
}



// ========================================
// QLUATEXTEDIT


class QLuaTextEdit::Private : public QObject
{
  Q_OBJECT
public:
  ~Private();
  Private(QLuaTextEdit *q);
  QString filterText(QString);
  void expandTab();
  bool eventFilter(QObject *watched, QEvent *event);
  static QTextCharFormat defaultFormat(QString);
  static void saveFormats();
public slots:
  void updateMatch();
  void updateHighlight();
  void scheduleLayout();
  void layout();
public:
  QLuaTextEdit *q;
  LineNumbers *lineNumbers;
  bool showLineNumbers;
  bool autoComplete;
  bool autoIndent;
  bool autoHighlight;
  bool autoMatch;
  bool tabExpand;
  int tabSize;
  QSize sizeInChars;
  bool layoutScheduled;
  QPointer<QLuaTextEditMode> mode;
  QPointer<QSyntaxHighlighter> highlighter;
};


QLuaTextEdit::Private::~Private()
{
  saveFormats();
  delete highlighter;
  delete mode;
}


QLuaTextEdit::Private::Private(QLuaTextEdit *q)
  : QObject(q), 
    q(q),
    lineNumbers(0),
    showLineNumbers(false),
    autoComplete(true), 
    autoIndent(true), 
    autoHighlight(true), 
    autoMatch(true),
    tabExpand(false), 
    tabSize(8),
    sizeInChars(QSize(80,25)),
    layoutScheduled(false)
{
  connect(q, SIGNAL(cursorPositionChanged()),
          this, SLOT(updateMatch()));
}


QString
QLuaTextEdit::Private::filterText(QString data)
{
  int pos = 0;
  int lastic = 0;
  QString dest;
  dest.reserve(data.size());
  for (int i=0; i<data.size(); i++)
    {
      QChar c = data.at(i);
      int ic = c.toAscii();
      if (ic == '\t')
        {
          int tpos = (int)((pos + tabSize) / tabSize) * tabSize;
          if (! tabExpand)
            dest += c;
          while (pos++ < tpos)
            if (tabExpand)
              dest += ' ';
        }
      else if (ic == '\n' || ic == '\r')
        {
          if (ic != '\n' || lastic != '\r')
            dest += '\n';
          pos = 0;
        }
      else if (isprint(ic) || isspace(ic) || c.isPrint() || c.isSpace())
        {
          dest += c;
          pos += 1;
        }
      lastic = ic;
    }
  dest.squeeze();
  return dest;
}


void
QLuaTextEdit::Private::expandTab()
{
  QTextCursor cursor = q->textCursor();
  cursor.beginEditBlock();
  if (cursor.hasSelection())
    cursor.deleteChar();
  int pos = cursor.position() - cursor.block().position();
  cursor.insertText(QString(tabSize - (pos % tabSize), QChar(' ')));
  cursor.endEditBlock();
}


void 
QLuaTextEdit::Private::updateMatch()
{
  bool b = false;;
  if (mode && mode->supportsMatch() && autoMatch)
    b = mode->doMatch();
  QList<QTextEdit::ExtraSelection> empty;
  if (!b && !q->extraSelections().isEmpty())
    q->setExtraSelections(empty);
}


void 
QLuaTextEdit::Private::updateHighlight()
{
  delete highlighter;
  if (mode && mode->supportsHighlight() && autoHighlight)
    highlighter = mode->highlighter();
  if (highlighter)
    QTimer::singleShot(0, highlighter, SLOT(rehighlight()));
}


bool 
QLuaTextEdit::Private::eventFilter(QObject *watched, QEvent *event)
{
  if (event->type() == QEvent::Resize 
      && watched == q->viewport() )
    scheduleLayout();
  return false;
}


void
QLuaTextEdit::Private::scheduleLayout()
{
  if (! layoutScheduled)
    QTimer::singleShot(0, this, SLOT(layout()));
  layoutScheduled = true;
}


void
QLuaTextEdit::Private::layout()
{
  layoutScheduled = false;
  if (! showLineNumbers)
    {
      if (lineNumbers)
        lineNumbers->hide();
      q->setViewportMargins(0, 0, 0, 0);
    }
  else
    {
      if (! lineNumbers)
        lineNumbers = new LineNumbers(q, q);
      int w = lineNumbers->sizeHint().width();
      q->setViewportMargins(w, 0, 0, 0);
      QRect r = q->viewport()->rect();
      r.setLeft(0);
      r.setWidth(w);
      lineNumbers->setGeometry(r);
      lineNumbers->show();
    }
}


QLuaTextEdit::QLuaTextEdit(QWidget *parent)
  : QPlainTextEdit(parent), d(new Private(this))
{
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  setLineWrapMode(NoWrap);
  viewport()->installEventFilter(d);
  connect(this, SIGNAL(blockCountChanged(int)), d, SLOT(scheduleLayout()));
  d->layout();
}


bool 
QLuaTextEdit::showLineNumbers() const
{
  return d->showLineNumbers;
}


bool 
QLuaTextEdit::autoComplete() const
{
  return d->autoComplete;
}


bool 
QLuaTextEdit::autoIndent() const
{
  return d->autoIndent;
}


bool 
QLuaTextEdit::autoHighlight() const
{
  return d->autoHighlight;
}


bool 
QLuaTextEdit::autoMatch() const
{
  return d->autoMatch;
}


bool 
QLuaTextEdit::tabExpand() const
{
  return d->tabExpand;
}


int 
QLuaTextEdit::tabSize() const
{
  return d->tabSize;
}


QSize 
QLuaTextEdit::sizeInChars() const
{
  return d->sizeInChars;
}


QLuaTextEditMode *
QLuaTextEdit::editorMode() const
{
  return d->mode;
}


void
QLuaTextEdit::setShowLineNumbers(bool b)
{
  if (d->showLineNumbers == b)
    return;
  d->showLineNumbers = b;
  d->scheduleLayout();
  emit settingsChanged();
}


void
QLuaTextEdit::setAutoComplete(bool b)
{
  if (d->autoComplete == b)
    return;
  d->autoComplete = b;
  emit settingsChanged();
}


void
QLuaTextEdit::setAutoIndent(bool b)
{
  if (d->autoIndent == b)
    return;
  d->autoIndent = b;
  emit settingsChanged();
}


void
QLuaTextEdit::setAutoHighlight(bool b)
{
  if (d->autoHighlight == b)
    return;
  d->autoHighlight = b;
  d->updateHighlight();
  emit settingsChanged();
}


void
QLuaTextEdit::setAutoMatch(bool b)
{
  if (d->autoMatch == b)
    return;
  d->autoMatch = b;
  d->updateMatch();
  emit settingsChanged();
}


void 
QLuaTextEdit::setTabExpand(bool b)
{
  if (d->tabExpand == b)
    return;
  d->tabExpand = b;
  emit settingsChanged();
}


void 
QLuaTextEdit::setTabSize(int s)
{
  if (s > 1 && s < 48 && s != d->tabSize)
    {
      d->tabSize = s;
      updateGeometry();
      emit settingsChanged();
    }
}


void 
QLuaTextEdit::setSizeInChars(QSize size)
{
  d->sizeInChars = size;
  updateGeometry();
}


bool
QLuaTextEdit::setEditorMode(QLuaTextEditModeFactory *factory)
{
  delete d->highlighter;
  delete d->mode;
  if (factory)
    d->mode = factory->create(this);
  d->updateMatch();
  d->updateHighlight();
  emit settingsChanged();
  return true;
}


bool 
QLuaTextEdit::setEditorMode(QString suffix)
{
  QLuaTextEditModeFactory *factory = 0;
  QList<QLuaTextEditModeFactory*> list = QLuaTextEditModeFactory::factories();
  foreach(QLuaTextEditModeFactory *f, list)
    if (f->suffixes().contains(suffix))
      factory = f;
  suffix = suffix.toLower();
  if (! factory)
    foreach(QLuaTextEditModeFactory *f, list)
      if (f->suffixes().contains(suffix))
        factory = f;
  if (factory)
    return setEditorMode(factory);
  return false;
}


bool 
QLuaTextEdit::readFile(QFile &file)
{
  if (file.open(QIODevice::ReadOnly))
    {
      QTextStream in(&file);
      QApplication::setOverrideCursor(Qt::WaitCursor);
      QString data = d->filterText(in.readAll());
      if (! file.error())
        {
          setPlainText(data);
          QApplication::restoreOverrideCursor();
          return true;
        }
      QApplication::restoreOverrideCursor();
    }
  return false;
}


bool 
QLuaTextEdit::readFile(QString fname)
{
  QFile file(fname);
  return readFile(file);
}


bool 
QLuaTextEdit::writeFile(QFile &file)
{
  if (file.open(QIODevice::WriteOnly))
    {
      QTextStream out(&file);
      QApplication::setOverrideCursor(Qt::WaitCursor);
      out << toPlainText();
      QApplication::restoreOverrideCursor();
      if (! file.error())
        return true;
    }
  return false;
}


bool 
QLuaTextEdit::writeFile(QString fname)
{
  QFile file(fname);
  return writeFile(file);
}


void 
QLuaTextEdit::showLine(int lineno)
{
  QTextDocument *d = document();
  QTextBlock block = d->findBlockByNumber(lineno-1);
  if (block.isValid())
    {
      window()->raise();
      window()->activateWindow();
      QTextCursor c(d);
      c.setPosition(getBlockIndent(block));
      setTextCursor(c);
      centerCursor();
      QList<QTextEdit::ExtraSelection> lextra;
      QTextEdit::ExtraSelection extra;
      c.setPosition(block.position()+block.length()-1, QTextCursor::KeepAnchor);
      extra.cursor = c;
      extra.format = format("(showline)/error");
      setExtraSelections(lextra << extra);
      return;
    }
  QApplication::beep();
}


bool
QLuaTextEdit::print(QPrinter *printer)
{
  QTextDocument *doc = 0;
  QTextDocument *odoc = document();
  QString title = odoc->metaInformation(QTextDocument::DocumentTitle);
  if (printer->printRange() == QPrinter::Selection)
    {
      // print selection
      doc = new QTextDocument(this);
      doc->setDefaultTextOption(odoc->defaultTextOption());
      doc->setMetaInformation(QTextDocument::DocumentTitle, title);
      doc->setDefaultFont(odoc->defaultFont());
      doc->setUseDesignMetrics(odoc->useDesignMetrics());
      QTextCursor cursor(doc);
      cursor.insertText(textCursor().selectedText());
    }
  else
    {
      // print document
      doc = odoc->clone(this);
      for (QTextBlock srcBlock = odoc->begin(), dstBlock = doc->begin();
           srcBlock.isValid() && dstBlock.isValid();
           srcBlock = srcBlock.next(), dstBlock = dstBlock.next())
        {
          QTextLayout *d = dstBlock.layout();
          QTextLayout *s = srcBlock.layout();
          d->setAdditionalFormats(s->additionalFormats());
        }
    }
  // make sure we wrap lines
  for (QTextBlock dstBlock = doc->begin(); 
       dstBlock.isValid(); 
       dstBlock = dstBlock.next())
    {
      QTextLayout *d = dstBlock.layout();
      QTextOption opt = d->textOption();
      opt.setWrapMode(QTextOption::WordWrap);
      d->setTextOption(opt);
    }
  // page layout
  QAbstractTextDocumentLayout *layout = doc->documentLayout();
  layout->setPaintDevice(printer);
  QSizeF size(printer->width(), printer->height());
  QTextFrameFormat fmt = doc->rootFrame()->frameFormat();
  int margin = printer->logicalDpiY();
  fmt.setMargin(margin);
  doc->rootFrame()->setFrameFormat(fmt);
  doc->setPageSize(size = size * 1.414);
  // prepare painter
  QPainter p(printer);
  const QSizeF pSize(printer->pageRect().size());
  p.scale(pSize.width()/size.width(), pSize.height()/size.height());
  int pageCopies = 1;
  int docCopies = printer->numCopies();
  if (printer->collateCopies())
    qSwap(docCopies, pageCopies);
  int fromPage = printer->fromPage();
  int toPage = printer->toPage();
  fromPage = (fromPage) ? qMax(1, fromPage) : 1;
  toPage = (toPage) ? qMin(doc->pageCount(), toPage) : doc->pageCount();
  int incPage = (printer->pageOrder() == QPrinter::LastPageFirst) ? -1 : 1;
  if (incPage < 0)
    qSwap(fromPage, toPage);
  int nPages = 0;
  QPrinter::PrinterState state = printer->printerState();
  for (int i = 0; i < docCopies; ++i)
    for (int page = fromPage; page != toPage + incPage; page += incPage)
      for (int j = 0; j < pageCopies; ++j) 
        {
          state = printer->printerState();
          if (state != QPrinter::Aborted && state != QPrinter::Error)
            {
              if (nPages++)
                printer->newPage();
              p.save();
              QPointF top(0, (page-1)*size.height());
              QRectF view(top, size);
              p.translate(-top);
              // print page
              QAbstractTextDocumentLayout::PaintContext ctx;
              p.setClipRect(view);
              ctx.clip = view;
              ctx.palette.setColor(QPalette::Text, Qt::black);
              layout->draw(&p, ctx);
              // print header
              QFont font = doc->defaultFont();
              font.setWeight(QFont::Bold);
              p.setFont(font);
              QFontMetrics m(font, printer);
              int h = m.lineSpacing();
              int w = (int)(view.width() * 0.67);
              QRectF header(margin, margin-h, view.width()-2*margin, h);
              header.translate(0, -h/3);
              QString f = m.elidedText(title,Qt::ElideLeft, w);
              p.translate(top);
              p.fillRect(header, QBrush(Qt::lightGray));
              p.drawText(header, Qt::AlignRight, 
                         QString("%1 ").arg(f));
              p.drawText(header, Qt::AlignLeft, 
                         QString(" Page %1").arg(page));
              // finish
              p.restore();
            }
        }
  // finish
  delete doc;
  return (state != QPrinter::Aborted) && (state != QPrinter::Error);
}


QSize 
QLuaTextEdit::sizeHint() const
{
  QFontMetrics fontMetrics(font());
  int cellw = fontMetrics.width("MM") - fontMetrics.width("M");
  int cellh = fontMetrics.lineSpacing();
  int w = d->sizeInChars.width();
  int h = d->sizeInChars.height();
  w += 4;
  if (d->showLineNumbers)
    w += 5;
  int t = cellw * d->tabSize;
  if (t != tabStopWidth())
    const_cast<QLuaTextEdit*>(this)->setTabStopWidth(t);
  return QSize(cellw * w, cellh * h);
}


void
QLuaTextEdit::keyPressEvent(QKeyEvent *event)
{
  // overwrite mode
  if (event->key() == Qt::Key_Insert &&
      event->modifiers() == Qt::NoModifier)
    {
      setOverwriteMode(! overwriteMode());
      emit settingsChanged();
      return;
    }
  // autoindent and autocomplete
  QString s = event->text();
  if (s == "\n" || s == "\r")
    {
      if (d->autoIndent && d->mode && d->mode->supportsIndent())
        d->mode->doEnter();
      else
        QPlainTextEdit::keyPressEvent(event);
      return;
    }
  else if (s == "\t")
    {
      bool ok = false;
      if (d->autoIndent)
        if (d->mode && d->mode->supportsIndent())
          ok = d->mode->doTab() || ok;
      if (d->autoComplete)
        if (d->mode && d->mode->supportsComplete())
          if (! textCursor().hasSelection())
            ok = d->mode->doComplete() || ok;
      if (! ok)
        QPlainTextEdit::keyPressEvent(event);
      return;
    }
  // default
  QPlainTextEdit::keyPressEvent(event);
}


int 
QLuaTextEdit::indentAt(int pos)
{
  QTextBlock block = document()->findBlock(pos);
  return indentAt(pos, block);
}


int 
QLuaTextEdit::indentAt(int pos, QTextBlock block)
{
  int c = 0;
  int bpos = block.position();
  if (block.isValid() && pos>=bpos)
    {
      int ts = d->tabSize;
      QString text = block.text();
      int ss = text.size();
      int e = pos-block.position();
      for (int i=0; i<e && i<ss; i++)
        if (text[i].toAscii() == '\t') 
          c = ((int)(c / ts) + 1) * ts;
        else
          c = c + 1;
    }
  return c;
}


int 
QLuaTextEdit::indentAfter(int pos, int dpos)
{
  QTextBlock block = document()->findBlock(pos);
  if (block.isValid())
    {
      QString text = block.text();
      int i = pos - block.position();
      for (; i< text.size(); i++)
        if (! text[i].isSpace())
          break;
      if (i < text.size())
        return indentAt(block.position() + i);
    }
  return indentAt(pos) + dpos;
}


int
QLuaTextEdit::getBlockIndent(QTextBlock block)
{
  int indent;
  return getBlockIndent(block, indent);
}


int
QLuaTextEdit::getBlockIndent(QTextBlock block, int &indent)
{
  
  Q_ASSERT(block.isValid());
  indent = -1;
  QString text = block.text();
  int ss = text.size();
  int ts = d->tabSize;
  int i;
  indent = 0;
  for (i=0; i<ss && text[i].isSpace(); i++)
    indent += (text[i].toAscii() == '\t') ? ts : 1;
  if (i >= ss)
    indent = -1;
  return block.position() + i;
}


int
QLuaTextEdit::setBlockIndent(QTextBlock block, int indent)
{
  Q_ASSERT(block.isValid());
  int oindent;
  int cpos = getBlockIndent(block, oindent);
  if (oindent == indent)
    return cpos;
  QTextCursor cursor(block);
  cursor.setPosition(cpos, QTextCursor::KeepAnchor);
  QString spaces;
  if (!d->tabExpand && d->tabSize > 0)
    for (; indent >= d->tabSize; indent -= d->tabSize)
      spaces += '\t';
  for(; indent>=1; indent-=1)
    spaces += ' ';
  cursor.insertText(spaces);
  return cursor.position();
}


QDialog*
QLuaTextEdit::makeGotoDialog()
{
  return new GotoDialog(this);
}


QDialog*
QLuaTextEdit::makeFindDialog()
{
  return new FindDialog(this);
}


QDialog*
QLuaTextEdit::makeReplaceDialog()
{
  return new ReplaceDialog(this);
}


void
QLuaTextEdit::prepareDialog(QDialog *d)
{
  GotoDialog *gd = qobject_cast<GotoDialog*>(d);
  if (gd)
    gd->prepare();
  FindDialog *fd = qobject_cast<FindDialog*>(d);
  if (fd)
    fd->prepare();
  ReplaceDialog *rd = qobject_cast<ReplaceDialog*>(d);
  if (rd)
    rd->prepare();
}


QRectF 
QLuaTextEdit::blockBoundingGeometry(const QTextBlock &block) const
{
  // just to make this function public!
  return QPlainTextEdit::blockBoundingGeometry(block);
}




// ========================================
// FORMATS


typedef QMap<QString,QTextCharFormat> Formats;


Q_GLOBAL_STATIC(Formats, formats);


QTextCharFormat 
QLuaTextEdit::Private::defaultFormat(QString key)
{
  QTextCharFormat fmt;
  if (key.endsWith("/comment")) {
    fmt.setFontItalic(true);
    fmt.setForeground(Qt::darkRed);
  } else if (key.endsWith("/string")) {
    fmt.setForeground(Qt::darkGray);
  } else if (key.endsWith("/keyword")) {
    fmt.setForeground(Qt::darkMagenta);
  } else if (key.endsWith("/cpp")) {
    fmt.setForeground(Qt::magenta);
  } else if (key.endsWith("/function")) {
    fmt.setForeground(Qt::darkCyan);
    fmt.setFontWeight(QFont::Bold);
  } else if (key.endsWith("/quote")) {
    fmt.setForeground(Qt::darkGreen);
  } else if (key.endsWith("/type")) {
    fmt.setForeground(Qt::darkGreen);
  } else if (key.endsWith("/url")) {
    fmt.setForeground(Qt::blue);
  } else if (key.endsWith("/error")) {
    fmt.setBackground(QColor(255,127,127));
  } else if (key.startsWith("(matcher)/")) {
    fmt.setBackground(QColor(240,240,64));
  }
  return fmt;
}


QTextCharFormat 
QLuaTextEdit::format(QString key)
{
  // quick cache
  if (formats()->contains(key))
    return (*formats())[key];
  // defaults
  QTextCharFormat fmt = Private::defaultFormat(key);
  // settings
  QSettings s;
  s.beginGroup("formats");
  s.beginGroup(key);
  if (s.contains("italic"))
    fmt.setFontItalic(s.value("italic").toBool());
  if (s.contains("weight")) {
    int i = s.value("weight").toInt();
    if ((i >= QFont::Light) && (i <= QFont::Black))
      fmt.setFontWeight(i);
  }
  if (s.contains("color")) {
    QColor c;
    c.setNamedColor(s.value("color").toString());
    if (c.isValid())
      fmt.setForeground(c);
  }
  if (s.contains("bgcolor")) {
    QColor c;
    c.setNamedColor(s.value("bgcolor").toString());
    if (c.isValid())
      fmt.setBackground(c);
  }
  // cache and return
  (*formats())[key] = fmt;
  return fmt;
}

void
QLuaTextEdit::setFormat(QString key, QTextCharFormat fmt)
{
  (*formats())[key] = fmt;
}

void 
QLuaTextEdit::Private::saveFormats()
{
  QSettings s;
  s.beginGroup("formats");
  Formats::const_iterator it = formats()->begin();
  for(; it != formats()->end(); ++it)
    {
      QString key = it.key();
      QTextCharFormat fmt = it.value();
      QTextCharFormat dfmt = defaultFormat(key);
      if (fmt != dfmt)
        {
          s.beginGroup(key);
          s.setValue("italic", fmt.fontItalic());
          s.setValue("weight", fmt.fontWeight());
          s.setValue("color", fmt.foreground().color());
          s.setValue("bgcolor", fmt.background().color());
          s.endGroup();
        }
    }
}


void 
QLuaTextEdit::reHighlight()
{
  formats()->clear();
  d->updateHighlight();
}



// ========================================
// MOC


#include "qluatextedit.moc"





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
