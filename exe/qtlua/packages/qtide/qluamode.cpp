/* -*- C++ -*- */

#include <QAbstractListModel>
#include <QApplication>
#include <QDebug>
#include <QDir>
#include <QDesktopWidget>
#include <QEventLoop>
#include <QFileInfo>
#include <QKeyEvent>
#include <QListView>
#include <QMap>
#include <QModelIndex>
#include <QMouseEvent>
#include <QPalette>
#include <QRegExp>
#include <QScrollBar>
#include <QVariant>


#include "qluamode.h"




// ========================================
// USER DATA


int 
QLuaModeUserData::highlightState()
{
  return -1;
}




// ========================================
// PRIVATE


struct Fmt { 
  int pos; 
  int len; 
  QTextCharFormat format; 
};


struct Mch {
  int pos;
  int len;
  int previous;
  bool more;
  bool error;
};


struct Bal {
  int pos;
  int len;
  bool outer;
};


struct Ind {
  int indent;
  int nindent;
  int noverlay;
};


class QLuaMode::Private : public QObject
{
  Q_OBJECT
public:
  QLuaMode *q;
  QMap<int,Fmt> formats;
  QMap<int,Mch> matches;
  QMap<int,Bal> balances;
  QMap<int,Ind> indents;
  int lastPos;
  // temporaries
  QMap<int,int> tmpIndents;
  QMap<int,int> tmpOverlays;

public:
  Private(QLuaMode *q) : QObject(q), q(q), lastPos(0) {}
  void setFormat(int pos, int len, const QTextCharFormat &format);
  int indent(const QTextBlock &block);
  void prepareParseBlock(const QTextBlock &block);
  void finishParseBlock(const QTextBlock &block);
public slots:
  void contentsChange(int pos, int removed, int added);
  void clearFrom(int pos);
  void parseTo(int pos);
};


void 
QLuaMode::Private::setFormat(int pos, int len, const QTextCharFormat &format)
{
  QMap<int,Fmt>::iterator it = formats.upperBound(pos);
  if (it != formats.end() && it.key() == pos)
    ++it;
  while (it != formats.end() && pos+len > it->pos)
    {
      if (pos > it->pos)
        {
          Fmt fmt = it.value();
          fmt.len = pos - it->pos;
          formats[fmt.len + fmt.pos] = fmt;
        }
      if (pos+len < it->pos + it->len)
        {
          int shift = pos + len - it->pos;
          it->pos += shift;
          it->len -= shift;
          ++it;
        }
      else
        {
          it = formats.erase(it);
        }
    }
  Fmt fmt;
  fmt.pos = pos;
  fmt.len = len;
  fmt.format = format;
  formats[pos + len] = fmt;
}



template<class T>
static void deleteAfter(T &m, int pos) 
{
  typename T::iterator it = m.lowerBound(pos);
  while (it != m.end())
    it = m.erase(it);
}


void 
QLuaMode::Private::contentsChange(int pos, int removed, int added)
{
  if (removed > 0 || added > 0)
    clearFrom(pos);
}


void 
QLuaMode::Private::clearFrom(int pos)
{
  if (pos < lastPos)
    {
      QTextBlock block = q->e->document()->findBlock(pos);
      int bpos = block.position();
      deleteAfter(formats, bpos);
      deleteAfter(matches, bpos);
      deleteAfter(balances, bpos);
      deleteAfter(indents, bpos);
      lastPos = bpos;
    }
}


void 
QLuaMode::Private::parseTo(int pos)
{
  if (lastPos <= pos)
    {
      QTextBlock block = q->e->document()->findBlock(lastPos);
      QTextBlock pblock = block.previous();
      QLuaModeUserData *idata = 0;
      QLuaModeUserData *odata = 0;
      if (pblock.isValid())
        idata = static_cast<QLuaModeUserData*>(pblock.userData());
      while (block.isValid() && block.position() <= pos)
        {
          int pos = block.position();
          prepareParseBlock(block);
          q->parseBlock(pos, block, idata, odata);
          finishParseBlock(block);
          block.setUserData(odata);
          block.setUserState(odata->highlightState());
          lastPos = pos + block.length() + 1;
          block = block.next();
          idata = odata;
        }
      if (block.isValid())
        lastPos = block.position();
    }
}


int
QLuaMode::Private::indent(const QTextBlock &block)
{
  Q_ASSERT(block.isValid());
  int indent = -1;
  int bpos = block.position();
  parseTo(bpos);
  if (indents.contains(bpos))
    indent = indents[bpos].indent;
  if (indent >= 0)
    return q->e->setBlockIndent(block, indent);
  return -1;
}


void 
QLuaMode::Private::prepareParseBlock(const QTextBlock &block)
{
  Q_ASSERT(block.isValid());
  int bpos = block.position();
  QTextBlock pblock = block.previous();
  int ppos = pblock.position();

  // indents
  tmpIndents.clear();
  tmpOverlays.clear();
  tmpIndents[-9999] = -1;
  tmpOverlays[-9999] = -2;
  QMap<int,Ind>::const_iterator it = indents.find(ppos);
  if (pblock.isValid() && it != indents.end())
    {
      tmpIndents[bpos-1] = it->nindent;
      tmpOverlays[bpos-1] = it->noverlay;
    }
}

void 
QLuaMode::Private::finishParseBlock(const QTextBlock &block)
{
  Q_ASSERT(block.isValid());
  int bpos = block.position();
  int cpos = q->e->getBlockIndent(block);

  // indents
  Ind ind;
  tmpIndents[-9999] = -1;
  tmpOverlays[-9999] = -2;
  QMap<int,int>::const_iterator iti = tmpIndents.end();
  ind.nindent = (--iti).value();
  QMap<int,int>::const_iterator ito = tmpOverlays.end();
  ind.noverlay = (--ito).value();
  iti = tmpIndents.lowerBound(cpos+1);
  ito = tmpOverlays.lowerBound(cpos+1);
  ind.indent = (--iti).value();
  if ((--ito).key() >= iti.key() && ito.value() > -2)
    ind.indent = ito.value();
  indents[bpos] = ind;
  tmpIndents.clear();
  tmpOverlays.clear();
}



// ========================================
// HIGHLIGHTER


class QLuaMode::Highlighter : public QSyntaxHighlighter
{
  Q_OBJECT
private:
  QLuaMode *m;
public:
  Highlighter(QLuaMode *m);
  virtual void highlightBlock(const QString &text);
};


QLuaMode::Highlighter::Highlighter(QLuaMode *m) 
  : QSyntaxHighlighter(m), m(m) 
{
  setDocument(m->e->document()); 
}


void 
QLuaMode::Highlighter::highlightBlock(const QString &text)
{
  QTextBlock block = currentBlock();
  int pos = block.position();
  int len = text.size();
  m->d->parseTo(pos);
  QMap<int,Fmt>::const_iterator it = m->d->formats.lowerBound(pos);
  for (; it != m->d->formats.end() && it->pos < pos+len; ++it)
    {
      int s = qMax(it->pos, pos);
      int e = qMin(it->pos + it->len, pos + len);
      setFormat(s-pos, e-s, it->format);
    }
}




// ========================================
// MODE



QLuaMode::QLuaMode(QLuaTextEditModeFactory *f, QLuaTextEdit *e)
  : QLuaTextEditMode(f, e),
    d(new Private(this))
{
  connect(e->document(), SIGNAL(contentsChange(int,int,int)),
          d, SLOT(contentsChange(int,int,int)) );
}


QSyntaxHighlighter *
QLuaMode::highlighter()
{
  return new Highlighter(this);
}


bool 
QLuaMode::doEnter()
{
  QTextCursor s = e->textCursor();
  s.insertBlock();
  if (s.block().previous().isValid())
    d->indent(s.block().previous());
  return true;
}


bool 
QLuaMode::doTab()
{
  int pos = 0;
  QTextCursor s = e->textCursor();
  QTextBlock b = e->document()->findBlock(s.selectionStart());
  while (b.isValid() && b.position() <= s.selectionEnd())
    {
      pos = d->indent(b);
      b = b.next();
    }
  d->parseTo(pos);
  if (pos > s.position() && !s.hasSelection())
    {
      s.setPosition(pos);
      e->setTextCursor(s);
    }
  return true;
}


bool 
QLuaMode::doMatch()
{
  QTextCursor c = e->textCursor();
  int spos = c.selectionStart();
  int epos = c.selectionEnd();
  d->parseTo(epos);
  QMap<int,Mch>::const_iterator mi = d->matches.lowerBound(epos);
  if (mi == d->matches.end() ||
      mi->pos + mi->len < epos ||
      mi->pos >= spos + (epos > spos) ? 1 : 0)
    return false;
  // find forward
  QMap<int,Mch>::const_iterator si = mi;
  QMap<int,Mch>::const_iterator ti = si;
  QTextBlock block = e->document()->findBlock(si->pos).next();
  while (ti != d->matches.end() && si->more)
    {
      if (ti->previous == si->pos + si->len)
        si = ti;
      QMap<int,Mch>::const_iterator tiplusone = ti; ++tiplusone;
      if (tiplusone == d->matches.end() 
          && block.isValid() && block.isVisible())
        {
          d->parseTo(block.position());
          block = block.next();
          continue;
        }
      ti = tiplusone;
    }
  // compute extra selections
  QTextCharFormat fmtMatch = e->format("(matcher)/ok");
  QTextCharFormat fmtError = e->format("(matcher)/error");
  QList<QTextEdit::ExtraSelection> matches;
  bool error = false;
  for(;;)
    {
      c.setPosition(si->pos);
      c.setPosition(si->pos + si->len, QTextCursor::KeepAnchor);
      QTextEdit::ExtraSelection match;
      match.cursor = c;
      if (si->error) {
        match.format = fmtError;
        error = true;
      } else
        match.format = fmtMatch;
      matches << match;
      if (si->previous < 0)
        break;
      si = d->matches.find(si->previous);
      if (si == d->matches.end())
        return false;
    }
  // finish
  if (matches.size() <= 1 && !error)
    return false;
  e->setExtraSelections(matches);
  return true;
}


bool 
QLuaMode::doBalance()
{
  QTextCursor c = e->textCursor();
  int spos = c.selectionStart();
  int epos = c.selectionEnd();
  d->parseTo(epos);
  QMap<int,Bal>::const_iterator mi = d->balances.lowerBound(epos);
  QTextBlock block = e->document()->findBlock(epos).next();
  while (mi == d->balances.end() && block.isValid())
    {
      d->parseTo(block.position());
      mi = d->balances.lowerBound(epos);
      block = block.next();
    }
  // find
  int nspos = -1;
  int nepos = epos;
  while (mi != d->balances.end() && nspos < 0)
    {
      nspos = -1;
      nepos = mi->pos + mi->len;
      while (mi != d->balances.end() && mi->pos + mi->len == nepos)
        {
          if (mi->pos < spos + (nepos > epos) ? 1 : 0)
            nspos = qMax(nspos, mi->pos);
          if (mi->outer && nspos < 0)
            return false;
          QMap<int,Bal>::const_iterator miplusone = mi; ++miplusone;
          if (miplusone == d->balances.end() && block.isValid())
            {
              d->parseTo(block.position());
              block = block.next();
              continue;
            }
          mi = miplusone;
        }
    }
  // select
  if (nspos < 0)
    return false;
  if ((nspos < spos && nepos >= epos) ||
      (nspos <= spos && nepos > epos) )
    {
      c.setPosition(nspos);
      c.setPosition(nepos, QTextCursor::KeepAnchor);
      e->setTextCursor(c);
      return true;
    }
  return false;
}


void 
QLuaMode::setFormat(int pos, int len, QString format)
{
  QString key = name() + "/" + format;
  setFormat(pos, len, e->format(key));
}


void 
QLuaMode::setFormat(int pos, int len, QTextCharFormat format)
{
  d->setFormat(pos, len, format);
}


void 
QLuaMode::setLeftMatch(int pos, int len)
{
  Mch mch;
  mch.pos = pos;
  mch.len = len;
  mch.previous = -1;
  mch.more = true;
  mch.error = false;
  d->matches[pos+len] = mch;
}


void 
QLuaMode::setMiddleMatch(int pos, int len, int ppos, int plen)
{
  Mch mch;
  mch.pos = pos;
  mch.len = len;
  mch.previous = ppos + plen;
  mch.more = true;
  mch.error = false;
  QMap<int,Mch>::iterator it = d->matches.find(mch.previous);
  if (it != d->matches.end() && it->more)
    d->matches[pos+len] = mch;
  else
    qWarning("QLuaMode::setMiddleMatch: broken match");
  
}


void 
QLuaMode::setRightMatch(int pos, int len, int ppos, int plen)
{
  Mch mch;
  mch.pos = pos;
  mch.len = len;
  mch.previous = ppos + plen;
  mch.more = false;
  mch.error = false;
  QMap<int,Mch>::iterator it = d->matches.find(mch.previous);
  if (it != d->matches.end() && it->more)
    d->matches[pos+len] = mch;
  else
    qWarning("QLuaMode::setRightMatch: broken match");
}


void 
QLuaMode::setErrorMatch(int pos, int len, int ppos, int plen, bool here)
{
  Mch mch;
  mch.pos = pos;
  mch.len = len;
  mch.previous = -1;
  mch.more = false;
  mch.error = true;
  int previous = ppos + plen;
  QMap<int,Mch>::iterator it = d->matches.find(previous);
  if (it != d->matches.end() && it->more)
    {
      mch.error = here;
      mch.previous = previous;
    }
  d->matches[pos+len] = mch;
  if (! here)
    for(; it != d->matches.end(); it = d->matches.find(it->previous))
      it->error = true;
}


int  
QLuaMode::followMatch(int pos, int len)
{
  int previous = pos + len;
  for (;;)
    {
      QMap<int,Mch>::iterator it = d->matches.find(previous);
      if (it == d->matches.end()) 
        return pos;
      pos = it->pos;
      len = it->len;
      previous = it->previous;
    }
}


void 
QLuaMode::setBalance(int fpos, int tpos)
{
  bool outer = true;
  QTextBlock block = e->document()->findBlock(tpos);
  if (block.isValid() && fpos > block.position())
    outer = false;
  setBalance(fpos, tpos, outer);
}


void 
QLuaMode::setBalance(int fpos, int tpos, bool outer)
{
  QMap<int,Bal>::const_iterator it = d->balances.lowerBound(tpos);
  for(; it != d->balances.end() && it.key() == tpos; ++it)
    if (it->pos == fpos)
      return;
  Bal bal;
  bal.pos = fpos;
  bal.len = tpos - fpos;
  bal.outer = outer;
  d->balances.insertMulti(tpos, bal);
}


void 
QLuaMode::setIndent(int pos, int indent)
{
  d->tmpIndents[pos] = indent;
}


void 
QLuaMode::setIndentOverlay(int pos, int indent)
{
  d->tmpOverlays[pos] = indent;
}




// ========================================
// COMPLETIONS



class 
QLuaMode::CompView : public QListView
{
  Q_OBJECT
public:
  CompView(QString s, QStringList c, QLuaTextEdit *p);
  int exec(QPoint pos);
public slots:
  void slotActivated(const QModelIndex &index);
protected:
  virtual bool event(QEvent *event);
public:
  class Model;
  QLuaTextEdit *e;
  CompModel *model;
  QString stem;
  QStringList completions;
  QEventLoop *loop;
  int selected;
};


class 
QLuaMode::CompModel : public QAbstractListModel
{
  Q_OBJECT
public:
  QLuaMode::CompView *q;
  CompModel(QLuaMode::CompView *parent);
  int rowCount(const QModelIndex &parent=QModelIndex()) const;
  QVariant data(const QModelIndex &index, int role) const;
  Qt::ItemFlags flags(const QModelIndex & index) const;
};


QLuaMode::CompModel::CompModel(QLuaMode::CompView *parent)
  : QAbstractListModel(parent), q(parent)
{
}


int 
QLuaMode::CompModel::rowCount(const QModelIndex &parent) const
{
  return q->completions.size();
}


QVariant 
QLuaMode::CompModel::data(const QModelIndex &index, int role) const
{
  int row = index.row();
  if (row >= 0 && row < q->completions.size())
    {
      switch(role)
        {
        case Qt::DisplayRole:
          return q->stem + q->completions[row] + QString::fromLatin1("\xA0");
        case Qt::FontRole:
          return q->font();
        case Qt::TextAlignmentRole:
          return Qt::AlignLeft;
        default:
          break;
        }
    }
  return QVariant();
}


Qt::ItemFlags
QLuaMode::CompModel::flags(const QModelIndex &index) const
{ 
  return Qt::ItemIsSelectable | Qt::ItemIsEnabled;
}


QLuaMode::CompView::CompView(QString s, QStringList c, QLuaTextEdit *p)
  : QListView(p),
    e(p),
    model(new CompModel(this)),
    stem(s),
    completions(c),
    loop(0)
{
  // activation
  connect(this, SIGNAL(clicked(const QModelIndex&)),
          this, SLOT(slotActivated(const QModelIndex&)) );
  connect(this, SIGNAL(activated(const QModelIndex&)),
          this, SLOT(slotActivated(const QModelIndex&)) );
  // setup
  setModel(model);
  setEditTriggers(NoEditTriggers);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setSelectionBehavior(SelectItems);
  setSelectionMode(SingleSelection);
  setViewMode(IconMode);
  setWrapping(true);
  setFlow(TopToBottom);
  setMovement(Static);
  setSpacing(0);
  setWindowFlags(Qt::Popup);
  setWindowModality(Qt::ApplicationModal);
  setBackgroundRole(QPalette::AlternateBase);
  viewport()->setBackgroundRole(QPalette::AlternateBase);
  setFrameStyle(Box|Plain);
  setLineWidth(2);
  // smaller font
  QFont font;
  const qreal fontFactor = 0.9;
  if (font.pixelSize() > 0)
    font.setPixelSize(qRound(font.pixelSize() * fontFactor + 0.5));
  else
    font.setPointSizeF(font.pointSizeF() * fontFactor);
  setFont(font);
}


int 
QLuaMode::CompView::exec(QPoint pos)
{
  QWidget *widget = e;
  // compute geometry
  ensurePolished();
  const int cmax = 10;
  const int count = model->rowCount();
  int rw = sizeHint().width();
  int rh = (sizeHintForRow(0) + spacing()) * qMax(3, qMin(cmax, count+1));
  resize(rw, rh);
  
  // estimate layout
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  int vh = maximumViewportSize().height();
  int vw = 8 + spacing();
  int sh = 0, sw = 0;
  for (int i = 0; i<count; i++) {
    QSize hint = sizeHintForIndex(model->index(i));
    if (sh + hint.height() + spacing() >= vh) 
      { vw += sw; sh = 0; sw = 0; }
    sh += hint.height() + spacing();
    sw = qMax(sw, hint.width() + spacing());
  }
  rw = qBound(rw / 2, vw+sw, e->width() * 2 / 3);
  if (rw < vw + sw)
    setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  // position list
  pos = widget->mapToGlobal(pos);
  QRect screen = QApplication::desktop()->availableGeometry(widget);
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


void 
QLuaMode::CompView::slotActivated(const QModelIndex &index)
{
  selected = -1;
  if (index.isValid() && index.row() >= 0 && index.row() < completions.size())
    selected = index.row();
  if (loop)
    loop->quit();
}


bool 
QLuaMode::CompView::event(QEvent *event)
{
  switch (event->type()) 
    {
    case QEvent::ShortcutOverride: 
      {
        switch( static_cast<QKeyEvent*>(event)->key())
          {
          case Qt::Key_Up: case Qt::Key_Down:
          case Qt::Key_Left: case Qt::Key_Right:
          case Qt::Key_Enter: case Qt::Key_Return: 
          case Qt::Key_Escape:
            event->accept();
            return true;
          default:
            break;
          }            
        break;
      }
    case QEvent::KeyPress: 
      {
        QKeyEvent *ke = static_cast<QKeyEvent*>(event);
        switch(ke->key())
          {
          default:
            QCoreApplication::postEvent(e, new QKeyEvent(*ke));
          case Qt::Key_Escape:
            emit activated(QModelIndex());
            event->accept();
            return true;
          case Qt::Key_Enter: 
          case Qt::Key_Return:
            emit activated(currentIndex());
            event->accept();
            return true;
          case Qt::Key_Tab:
          case Qt::Key_Up: case Qt::Key_Down:
          case Qt::Key_Left: case Qt::Key_Right:
            break;
          }
        break;
      }
    case QEvent::MouseButtonPress:
      {
        QPoint pos = static_cast<QMouseEvent*>(event)->globalPos();
        if (! rect().contains(mapFromGlobal(pos))) 
          {
            event->accept();
            selected = -1;
            if (loop)
              loop->quit();
            return true;
          }
        break;
      }
    default:
      break;
    }
  return QListView::event(event);
}


int 
QLuaMode::askCompletion(QString stem, QStringList completions)
{
  CompView *view = new CompView(stem, completions, e);
  int selected = view->exec(e->cursorRect().bottomRight());
  delete view;
  e->activateWindow();
  return selected;
}


void 
QLuaMode::fileCompletion(QString &stem, QStringList &completions)
{
  QFileInfo info(stem.isEmpty() ? "./" : stem);
  QString f = info.fileName();
  QDir d = info.dir();
  stem = f;
  foreach(QString h, d.entryList())
    {
      if (! h.startsWith(f))
        continue;
      if (QFileInfo(d,h).isDir())
        h += "/";
      completions += h.mid(f.size());
    }
}





// ========================================
// MOC


#include "qluamode.moc"





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
