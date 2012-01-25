/* -*- C++ -*- */

#define _USE_MATH_DEFINES
#include <math.h>

#include "qtluapainter.h"
#include "qtluaprinter.h"

#include <QAbstractTextDocumentLayout>
#include <QBrush>
#include <QColor>
#include <QColormap>
#include <QCoreApplication>
#include <QDebug>
#include <QEvent>
#include <QFont>
#include <QFontMetricsF>
#include <QGradient>
#include <QImage>
#include <QMainWindow>
#include <QMetaEnum>
#include <QMetaObject>
#include <QMetaType>
#include <QMutex>
#include <QMutexLocker>
#include <QObject>
#include <QPaintEngine>
#include <QPaintEvent>
#include <QPen>
#include <QPixmap>
#include <QRegion>
#include <QResizeEvent>
#include <QStack>
#include <QTransform>
#include <QTextBlockFormat>
#include <QTextCharFormat>
#include <QTextDocument>
#include <QTextFrame>
#include <QTextFrameFormat>
#include <QThread>
#include <QVector>



// ========================================
// QTLUAPAINTER::STATE


struct QtLuaPainter::State 
{
  bool hasPoint;
  QPointF point;
  QPainterPath path;
  QPainterPath clip;
  QTransform matrix;
  QBrush brush;
  QPen pen;
  QFont font;
  QBrush background;
  CompositionMode mode;
  RenderHints hints;
  AngleUnit unit;
  QString styleSheet;
public:
  State();
  void apply(QPainter *p);
  void transform(QTransform m, bool combine);
};


QtLuaPainter::State::State()
  :  hasPoint(false),
     brush(Qt::black),
     pen(Qt::black, 1, Qt::SolidLine, Qt::FlatCap, Qt::MiterJoin),
     mode(SourceOver),
     hints(Antialiasing|TextAntialiasing),
     unit(Degrees)
{
  pen.setMiterLimit(10);
  font.setStyleHint(QFont::SansSerif);
  font.setPixelSize(10);
}


void 
QtLuaPainter::State::apply(QPainter *p)
{
  if (p && p->isActive())
    {
      p->setWorldTransform(matrix, false);
      p->setBrush(brush);
      p->setPen(pen);
      p->setFont(font);
      if  (clip.isEmpty()) {
        p->setClipPath(clip, Qt::NoClip);
        p->setClipping(false);
      } else
        p->setClipPath(clip, Qt::ReplaceClip);
      if (background.style() == Qt::NoBrush)
        p->setBackgroundMode(Qt::TransparentMode);
      else
        p->setBackgroundMode(Qt::OpaqueMode);        
      p->setBackground(background);
      if (p->paintEngine()->hasFeature(QPaintEngine::PorterDuff))
        p->setCompositionMode(QPainter::CompositionMode((int)mode));
      p->setRenderHints(QPainter::RenderHints((int)hints));
    }
}


void 
QtLuaPainter::State::transform(QTransform m, bool combine)
{
  bool invertible;
  QTransform i = m.inverted(&invertible);
  if (combine)
    m = m * matrix;
  else
    i = matrix * i;
  matrix = m;
  point = (invertible) ? i.map(point) : QPointF();
  path = (invertible) ? i.map(path) : QPainterPath();
  clip = (invertible) ? i.map(clip) : QPainterPath();
}




// ========================================
// QTLUAPAINTER::PRIVATE


struct QtLuaPainter::Private : public QObject
{
  Q_OBJECT
public:
  ~Private();
  Private(QtLuaPainter *parent);
  void damage(QRectF r);
  void resize(int w, int h, bool monochrome);
  void repaint(QPaintEvent *e);
  void protect(QVariant v);
  void protect(const QBrush &b);
  void protect(const QPen &b);
public slots:
  void destroyed(QObject *obj);
protected:
  bool eventFilter(QObject *watcher, QEvent *event);
  bool event(QEvent *e);
public:
  QMutex mutex;
  QtLuaPainter *q;
  QPainter *p;
  State state;
  QStack<State> stack;
  QPaintDevice *device;
  QPrinter *printer;
  QImage image;
  QPixmap pixmap;
  QPointer<QWidget> widget;
  QPointer<QObject> object;
  QVariantList saved;
  QRegion damaged;
  int count;
  bool eject;
};


QtLuaPainter::Private::~Private()
{
  delete p;
}


QtLuaPainter::Private::Private(QtLuaPainter *parent)
  : QObject(parent),
    mutex(QMutex::Recursive),
    q(parent),
    p(0),
    device(0),
    printer(0),
    count(0),
    eject(false)
{
}


void
QtLuaPainter::Private::damage(QRectF r)
{
  r = state.matrix.mapRect(r);
  int x1 = qMax((int)floor(qMin(r.left(),r.right()))-1, 0);
  int x2 = (int)ceil(qMax(r.left(),r.right()))+1;
  int y1 = qMax((int)floor(qMin(r.top(),r.bottom()))-1, 0);
  int y2 = (int)ceil(qMax(r.top(),r.bottom()))+1;
  QRect ri(x1,y1,x2-x1,y2-y1);
  damaged |= ri;
}


void
QtLuaPainter::Private::destroyed(QObject *obj)
{
  if (obj == widget && device != static_cast<QPaintDevice*>(widget))
    return;
  if (obj == object)
    resize(0,0,false);
}


void
QtLuaPainter::Private::resize(int w, int h, bool monochrome)
{
  QMutexLocker lock(&mutex);
  QPixmap opixmap = pixmap;
  QImage oimage = image;
  // delete painter
  for (int i=0; i<stack.size(); i++)
    p->restore();
  delete p;
  p = 0;
  // create new pixmap/image
  if (w<=0 || h<=0)
    image = QImage();
  else if (monochrome)
    image = QImage(w,h,QImage::Format_Mono);
  else
    image = QImage(w, h, QImage::Format_ARGB32_Premultiplied);
  pixmap = QPixmap();
  // attach painter
  printer = 0;
  device = &image;
  if (image.isNull())
    p = new QPainter();
  else
    p = new QPainter(&image);
  QRect rect = image.rect();
  p->fillRect(rect, Qt::white);
  // copy old pixmap/image data
  if (p->isActive())
    {
      if (! oimage.isNull())
        {
          rect = rect.intersected(oimage.rect());
          p->drawImage(rect,oimage,rect);
        }
      else if (! opixmap.isNull())
        {
          rect = rect.intersected(opixmap.rect());
          p->drawPixmap(rect,opixmap,rect);
        }
      // restore
      for (int i=0; i<stack.size(); i++)
        {
          stack[i].apply(p);
          p->save();
        }
      state.apply(p);
    }
  // refresh
  q->gbegin();
  damaged = image.rect();
  q->gend();
}


void
QtLuaPainter::Private::repaint(QPaintEvent *e)
{
  if (device == &image && widget)
    {
      QMutexLocker lock(&mutex);
      QRect rect = e->rect();
      QPainter painter(widget);
      painter.drawImage(rect, image, rect);
    }
}


bool
QtLuaPainter::Private::eventFilter(QObject *watched, QEvent *event)
{
  if (watched && watched == widget && device == &image)
    {
      if (event->type() == QEvent::Resize) 
        {
          QSize s = static_cast<QResizeEvent*>(event)->size();
          resize(s.width(), s.height(), false);
        }
      else if (event->type() == QEvent::User)
        {
          QMutexLocker lock(&mutex);
          if (widget && !damaged.isEmpty())
            widget->update(damaged);
          damaged = QRegion();
        }
      else if (event->type() == QEvent::Paint)
        {
          repaint(static_cast<QPaintEvent*>(event));
        }
    }
  return false;
}


void 
QtLuaPainter::Private::protect(QVariant v)
{
  QMutexLocker lock(&mutex);
  int size = saved.size();
  saved.append(v);
  if (! size)
    QCoreApplication::postEvent(this, new QEvent(QEvent::User));
}


void 
QtLuaPainter::Private::protect(const QBrush &b)
{
  if (b.style() == Qt::TexturePattern)
    protect(QVariant(b));
}


void 
QtLuaPainter::Private::protect(const QPen &b)
{
  if (b.brush().style() == Qt::TexturePattern)
    protect(QVariant(b));
}


bool
QtLuaPainter::Private::event(QEvent *e)
{
  if (e->type() == QEvent::User)
    {
      QMutexLocker lock(&mutex);
      saved.clear(); // actual deletion of brushes and pens
      return true;
    }
  return QObject::event(e);
}



// ========================================
// QTLUAPAINTER::LOCKER


struct QtLuaPainter::Locker 
{
  QtLuaPainter *r;
  Locker(QtLuaPainter *r) : r(r) { r->d->mutex.lock(); r->gbegin(); }
  ~Locker() { if (r) { r->gend(); r->d->mutex.unlock(); } }
};



// ========================================
// QTLUAPAINTER


QtLuaPainter::~QtLuaPainter()
{
  while (d && d->stack.size() > 0) 
    grestore();
  while (d && d->count > 0) 
    gend();
  delete d;
}


QtLuaPainter::QtLuaPainter()
  : d(new Private(this))
{
  if (QThread::currentThread() != QCoreApplication::instance()->thread())
    qWarning("QtLuaPainter should be created from the main thread.");
  d->resize(0,0,false);
}


QtLuaPainter::QtLuaPainter(QWidget *widget, bool buffered)
  : QObject(widget), d(new Private(this))
{
  if (buffered)
    {
      QSize s = widget->size();
      d->object = widget;
      d->widget = widget;
      d->resize(s.width(), s.height(), false);
      widget->installEventFilter(d);
      connect(widget, SIGNAL(destroyed(QObject*)),
              d, SLOT(destroyed(QObject*)),
              Qt::DirectConnection);
    }
  else
    {
      d->object = widget;
      d->widget = widget;
      d->device = widget;
      d->p = new QPainter(widget);
    }
}


QtLuaPainter::QtLuaPainter(QObject *object)
  : QObject(object), d(new Private(this))
{
  QPrinter *printer = 0;
  QPaintDevice *device = 0;
  if (object)
    {
      const QMetaObject *mo = object->metaObject();
      mo->invokeMethod(object, "printer", Q_RETURN_ARG(QPrinter*, printer));
      mo->invokeMethod(object, "device", Q_RETURN_ARG(QPaintDevice*, device));
      d->object = object;
      d->widget = qobject_cast<QWidget*>(object);
      if (mo->indexOfSignal(SIGNAL(closing(QObject*))) >= 0)
        connect(object, SIGNAL(closing(QObject*)),
                d, SLOT(destroyed(QObject*)),
                Qt::DirectConnection);
      connect(object, SIGNAL(destroyed(QObject*)),
              d, SLOT(destroyed(QObject*)),
              Qt::DirectConnection);
    }
  if (device)
    {
      d->device = device;
      d->printer = printer;
      d->p = new QPainter(device);
      initmatrix();
    }
  else
    {
      d->resize(0,0,false);
    }
}


QtLuaPainter::QtLuaPainter(QImage image)
  : d(new Private(this))
{
  d->image = image;
  d->device = &d->image;
  d->p = new QPainter(d->device);
}


QtLuaPainter::QtLuaPainter(QPixmap pixmap)
  : d(new Private(this))
{
  d->pixmap = pixmap;
  d->device = &d->pixmap;
  d->p = new QPainter(d->device);
}


QtLuaPainter::QtLuaPainter(QString filename, const char *format)
  : d(new Private(this))
{
  QImage image(filename, format);
  if (! image.isNull())
    if (image.format() != QImage::Format_Mono &&
        image.format() != QImage::Format_ARGB32_Premultiplied )
      image = image.convertToFormat(QImage::Format_ARGB32_Premultiplied);
  d->image = image;
  d->device = &d->image;
  if (d->image.isNull())
    d->p = new QPainter();
  else
    d->p = new QPainter(&d->image);
}


QtLuaPainter::QtLuaPainter(int w, int h, bool monochrome)
  : d(new Private(this))
{
  d->resize(w,h, monochrome);
}


QWidget *
QtLuaPainter::widget() const
{
  return d->widget;
}


QObject*
QtLuaPainter::object() const
{
  return d->object;
}


QPrinter*
QtLuaPainter::printer() const
{
  return d->printer;
}


QImage 
QtLuaPainter::image() const
{
  if (d->device == &d->pixmap)
    return d->pixmap.toImage();
  else if (d->device == (QWidget*)(d->widget))
    return QPixmap::grabWindow(d->widget->winId()).toImage();
  else
    return d->image;
}


QPixmap 
QtLuaPainter::pixmap() const
{
  if (d->device == &d->image)
    return QPixmap::fromImage(d->image);
  else if (d->device == (QWidget*)(d->widget))
    return QPixmap::grabWindow(d->widget->winId());
  else
    return d->pixmap;
}


QPaintDevice *
QtLuaPainter::device() const
{
  return &d->image;
}


QPainter *
QtLuaPainter::painter() const
{
  return d->p;
}


QRect
QtLuaPainter::rect() const
{
  QtLuaPrinter *printer = qobject_cast<QtLuaPrinter*>(d->object);
  if (d->device == static_cast<QPaintDevice*>(printer))
    {
      QSizeF sz = printer->paperSize();
      if (sz.isValid())
        return QRect(0,0,(int)ceil(sz.width()),(int)ceil(sz.height()));
    }
  return QRect(0,0,d->device->width(),d->device->height());
}


QSize
QtLuaPainter::size() const
{
  QtLuaPrinter *printer = qobject_cast<QtLuaPrinter*>(d->object);
  if (d->device == static_cast<QPaintDevice*>(printer))
    {
      QSizeF sz = printer->paperSize();
      if (sz.isValid())
        return QSize((int)ceil(sz.width()),(int)ceil(sz.height()));
    }
  return QSize(d->device->width(),d->device->height());
}


int
QtLuaPainter::depth() const
{
  return d->device->depth();
}


void
QtLuaPainter::close()
{
  return d->resize(0,0,false);
}


void 
QtLuaPainter::gbegin()
{
  d->count += 1;
  // pending showpage
  if (d->printer && d->eject)
    d->printer->newPage();
  d->eject = false;
}


void 
QtLuaPainter::refresh()
{
  if (! d->damaged.isEmpty() && d->device == &d->image && d->widget)
    {
      QEvent *ev = new QEvent(QEvent::User);
      QCoreApplication::postEvent(d->widget, ev);
    }
}


void 
QtLuaPainter::gend(bool invalidate)
{
  if (invalidate)
    d->damaged = rect();
  d->count -= 1;
  if (d->count <= 0 && d->p)
    {
      refresh();
      d->count = 0;
    }
}


void 
QtLuaPainter::showpage()
{
  Locker lock(this);
  if (d->printer)
    {
      d->eject = true;
    }
  else
    {
      QRect rect = this->rect();
      d->p->save();
      d->p->resetTransform();
      d->p->fillRect(rect, Qt::white);
      d->p->restore();
      d->damaged = rect;
    }
}


QPen 
QtLuaPainter::currentpen() const
{
  return d->state.pen;
}

QBrush 
QtLuaPainter::currentbrush() const
{
  return d->state.brush;
}


QPointF 
QtLuaPainter::currentpoint() const
{
  return d->state.point;
}


QPainterPath 
QtLuaPainter::currentpath() const
{
  return d->state.path;
}


QPainterPath 
QtLuaPainter::currentclip() const
{
  return d->state.clip;
}


QFont 
QtLuaPainter::currentfont() const
{
  return d->state.font;
}


QTransform 
QtLuaPainter::currentmatrix() const
{
  return d->state.matrix;
}


QBrush 
QtLuaPainter::currentbackground() const
{
  return d->state.background;
}


QtLuaPainter::CompositionMode 
QtLuaPainter::currentmode() const
{
  return d->state.mode;
}


QtLuaPainter::RenderHints 
QtLuaPainter::currenthints() const
{
  return d->state.hints;
}


QtLuaPainter::AngleUnit
QtLuaPainter::currentangleunit() const
{
  return d->state.unit;
}


QString
QtLuaPainter::currentstylesheet() const
{
  return d->state.styleSheet;
}


void 
QtLuaPainter::setpen(QPen pen)
{
  QMutexLocker lock(&d->mutex);
  d->protect(d->state.pen);
  d->state.pen = pen;
  d->p->setPen(pen);
}


void 
QtLuaPainter::setbrush(QBrush brush)
{
  QMutexLocker lock(&d->mutex);
  d->protect(d->state.brush);
  d->state.brush = brush;
  d->p->setBrush(brush);
}


void 
QtLuaPainter::setpoint(QPointF p)
{
  d->state.point = p;
}


void 
QtLuaPainter::setpath(QPainterPath p)
{
  d->state.path = p;
}


void 
QtLuaPainter::setclip(QPainterPath p)
{
  QMutexLocker lock(&d->mutex);
  d->state.clip = p;
  d->p->setClipPath(p, (p.isEmpty()) ? Qt::NoClip : Qt::ReplaceClip);
}


void 
QtLuaPainter::setfont(QFont f)
{
  QMutexLocker lock(&d->mutex);
  d->state.font = f;
  d->p->setFont(f);
}


void 
QtLuaPainter::setmatrix(QTransform m)
{
  QMutexLocker lock(&d->mutex);
  d->state.transform(m, false);
  d->p->setWorldTransform(d->state.matrix, false);
  d->state.clip = d->p->clipPath();
}


void 
QtLuaPainter::setbackground(QBrush brush)
{
  Qt::BGMode bm;
  bm = (brush.style() == Qt::NoBrush) ? Qt::TransparentMode: Qt::OpaqueMode;
  QMutexLocker lock(&d->mutex);
  d->state.background = brush;
  d->p->setBackgroundMode(bm);
  d->p->setBackground(brush);
}


void 
QtLuaPainter::setmode(CompositionMode m)
{
  QMutexLocker lock(&d->mutex);
  d->state.mode = m;
  d->p->setCompositionMode(QPainter::CompositionMode(m));
}


void 
QtLuaPainter::sethints(RenderHints h)
{
  QMutexLocker lock(&d->mutex);
  d->state.hints = h;
  d->p->setRenderHints(QPainter::RenderHints((int)h));
}


void 
QtLuaPainter::setangleunit(AngleUnit u)
{
  d->state.unit = u;
}


void
QtLuaPainter::setstylesheet(QString s)
{
  d->state.styleSheet = s;
}


void 
QtLuaPainter::initclip()
{
  QMutexLocker lock(&d->mutex);
  d->state.clip = QPainterPath();
  d->p->setClipPath(d->state.clip, Qt::NoClip);
  d->p->setClipping(false);
}


void 
QtLuaPainter::initmatrix()
{
  QTransform init;
  QtLuaPrinter *printer = qobject_cast<QtLuaPrinter*>(d->object);
  if (d->device == static_cast<QPaintDevice*>(printer))
    {
      QSizeF ps = printer->paperSize();   
      if (ps.isValid())
        {
          QRect pr = printer->pageRect();
          qreal sx = ps.width() / pr.width();
          qreal sy = ps.height() / pr.height();
          if (sx > 0 || sy > 0)
            {
              qreal q = 1.0 / qMax(sx,sy);
              init.scale(q, q);
            }
        }
    }
  setmatrix(init);
}


void 
QtLuaPainter::initgraphics()
{
  d->state = State();
  d->state.apply(d->p);
}


void 
QtLuaPainter::scale(qreal x, qreal y)
{
  QTransform t;
  t.scale(x,y);
  QMutexLocker lock(&d->mutex);
  d->state.transform(t, true);
  d->p->setWorldTransform(d->state.matrix, false);
  d->state.clip = d->p->clipPath();
}


void 
QtLuaPainter::rotate(qreal x)
{
  QTransform t;
  if (d->state.unit == Radians)
    x *= 180 / M_PI;
  t.rotate(x);
  QMutexLocker lock(&d->mutex);
  d->state.transform(t, true);
  d->p->setWorldTransform(d->state.matrix, false);
  d->state.clip = d->p->clipPath();
}


void 
QtLuaPainter::translate(qreal x, qreal y)
{
  QTransform t;
  t.translate(x,y);
  QMutexLocker lock(&d->mutex);
  d->state.transform(t, true);
  d->p->setWorldTransform(d->state.matrix, false);
  d->state.clip = d->p->clipPath();
}


void 
QtLuaPainter::concat(QTransform m)
{
  QMutexLocker lock(&d->mutex);
  d->state.transform(m, true);
  d->p->setWorldTransform(d->state.matrix, false);
  d->state.clip = d->p->clipPath();
}


void 
QtLuaPainter::gsave()
{
  QMutexLocker lock(&d->mutex);
  d->stack.push(d->state);
  if (d->p->isActive())
    d->p->save();
}


void 
QtLuaPainter::grestore()
{
  QMutexLocker lock(&d->mutex);
  if (d->stack.size() > 0)
    {
      d->protect(d->state.pen);
      d->protect(d->state.brush);
      d->state = d->stack.pop();
      if (d->p->isActive())
        {
          d->p->restore();
          d->state.apply(d->p);
        }
    }
}

  
void 
QtLuaPainter::newpath()
{
  d->state.path = QPainterPath();
  d->state.hasPoint = false;
}


void 
QtLuaPainter::moveto(qreal x, qreal y)
{
  d->state.point = QPointF(x,y);
  d->state.path.moveTo(d->state.point);
  d->state.hasPoint = true;
}


void 
QtLuaPainter::lineto(qreal x, qreal y)
{
  if (! d->state.hasPoint)
    d->state.path.moveTo(d->state.point);
  d->state.point = QPointF(x,y);
  d->state.path.lineTo(d->state.point);
  d->state.hasPoint = true;
}


void 
QtLuaPainter::curveto(qreal x1, qreal y1, qreal x2, qreal y2, qreal x3, qreal y3)
{
  if (! d->state.hasPoint)
    d->state.path.moveTo(d->state.point);
  QPointF c1(x1,y1);
  QPointF c2(x2,y2);
  d->state.point = QPointF(x3,y3);
  d->state.path.cubicTo(c1,c2,d->state.point);
  d->state.hasPoint = true;
}


void 
QtLuaPainter::arc(qreal x, qreal y, qreal r, qreal a1, qreal a2)
{
  QRectF rect(x-r, y-r, 2*r, 2*r);
  if (d->state.unit == Radians)
    {
      a1 *= 180 / M_PI;
      a2 *= 180 / M_PI;
    }
  if (a2 < a1)
    a2 += 360 * ceil((a1-a2)/360);
  if (!d->state.hasPoint)
    d->state.path.arcMoveTo(rect, -a1);
  d->state.path.arcTo(rect, -a1, a1-a2);
  d->state.point = d->state.path.currentPosition();
  d->state.hasPoint = true;
}


void 
QtLuaPainter::arcn(qreal x, qreal y, qreal r, qreal a1, qreal a2)
{
  QRectF rect(x-r, y-r, 2*r, 2*r);
  if (d->state.unit == Radians)
    {
      a1 *= 180 / M_PI;
      a2 *= 180 / M_PI;
    }
  if (a2 > a1)
    a2 -= 360 * ceil((a2-a1)/360);
  if (!d->state.hasPoint)
    d->state.path.arcMoveTo(rect, -a1);
  d->state.path.arcTo(rect, -a1, a1-a2);
  d->state.point = d->state.path.currentPosition();
  d->state.hasPoint = true;
}


void 
QtLuaPainter::arcto(qreal x1, qreal y1, qreal x2, qreal y2, qreal r)
{
  if (! d->state.hasPoint)
    d->state.path.moveTo(d->state.point);
  QPointF p1(x1,y1);
  QPointF u = d->state.point - p1;
  QPointF v = QPointF(x2,y2) - p1;
  qreal s = u.x() * v.y() - u.y() * v.x();
  if (s == 0)
    {
      lineto(x1,y1);
    }
  else
    {
      qreal ulen = sqrt(u.x() * u.x() + u.y() * u.y()); 
      qreal vlen = sqrt(v.x() * v.x() + v.y() * v.y()); 
      u /= ulen;
      v /= vlen;
      s /= ulen * vlen;
      qreal c = u.x() * v.x() + u.y() * v.y();
      qreal aw = atan2(s, -c);
      qreal as = (s>0) ? atan2(-u.x(),u.y()) : atan2(u.x(),-u.y());
      aw *= 180.0 / M_PI;
      as *= 180.0 / M_PI;
      QPointF o = p1 + (u + v) * fabs(r / s);
      QRectF rect(o.x()-r, o.y()-r, 2*r, 2*r);
      d->state.path.arcTo(rect, -as, aw);
      d->state.point = d->state.path.currentPosition();
      d->state.hasPoint = true;
    }
}


void 
QtLuaPainter::rmoveto(qreal x, qreal y)
{
  d->state.point += QPointF(x,y);
  d->state.path.moveTo(d->state.point);
  d->state.hasPoint = true;
}


void 
QtLuaPainter::rlineto(qreal x, qreal y)
{
  if (! d->state.hasPoint)
    d->state.path.moveTo(d->state.point);
  d->state.point += QPointF(x,y);
  d->state.path.lineTo(d->state.point);
  d->state.hasPoint = true;
}


void 
QtLuaPainter::rcurveto(qreal x1, qreal y1, qreal x2, qreal y2, qreal x3, qreal y3)
{
  if (! d->state.hasPoint)
    d->state.path.moveTo(d->state.point);
  QPointF c1(x1,y1);
  QPointF c2(x2,y2);
  c1 += d->state.point;
  c2 += d->state.point;
  d->state.point += QPointF(x3,y3);
  d->state.path.cubicTo(c1,c2,d->state.point);
  d->state.hasPoint = true;
}


void
QtLuaPainter::rectangle(qreal x, qreal y, qreal w, qreal h)
{
  QRectF rect(x,y,w,h);
  d->state.path.addRect(rect);
  d->state.hasPoint = false;
}


void 
QtLuaPainter::charpath(QString text)
{
  QFontMetricsF metrics(d->state.font);
  double width = metrics.width(text);
  d->state.path.addText(d->state.point, d->state.font, text);
  d->state.point.rx() += width;
  d->state.hasPoint = false;
}


void 
QtLuaPainter::closepath()
{
  d->state.path.closeSubpath();
  d->state.point = QPointF(0,0);
  d->state.hasPoint = false;
}


void 
QtLuaPainter::stroke(bool resetpath)
{
  if (! d->state.path.isEmpty())
    { 
      Locker lock(this);
      d->p->strokePath(d->state.path, d->state.pen);
      qreal w = d->state.pen.width();
      d->damage(d->state.path.boundingRect().adjusted(-w,-w,w,w));
    }
  if (resetpath)
    newpath();
}


void 
QtLuaPainter::fill(bool resetpath)
{
  if (! d->state.path.isEmpty())
    { 
      Locker lock(this);
      d->state.path.setFillRule(Qt::WindingFill);
      d->p->fillPath(d->state.path, d->state.brush);
      d->damage(d->state.path.boundingRect());
    }
  if (resetpath)
    newpath();
}


void 
QtLuaPainter::eofill(bool resetpath)
{
  if (! d->state.path.isEmpty())
    { 
      Locker lock(this);
      d->state.path.setFillRule(Qt::OddEvenFill);
      d->p->fillPath(d->state.path, d->state.brush);
      d->damage(d->state.path.boundingRect());
    }
  if (resetpath)
    newpath();
}


void 
QtLuaPainter::clip(bool resetpath)
{
  QMutexLocker lock(&d->mutex);
  d->state.path.setFillRule(Qt::WindingFill);
  d->p->setClipPath(d->state.path, Qt::IntersectClip);
  d->state.clip = d->p->clipPath();
  if (resetpath)
    newpath();
}


void 
QtLuaPainter::eoclip(bool resetpath)
{
  QMutexLocker lock(&d->mutex);
  d->state.path.setFillRule(Qt::OddEvenFill);
  d->p->setClipPath(d->state.path, Qt::IntersectClip);
  d->state.clip = d->p->clipPath();
  if (resetpath)
    newpath();
}


void 
QtLuaPainter::show(QString text)
{
  Locker lock(this);
  QFontMetricsF metrics(d->state.font);
  double width = metrics.width(text);
  QRectF brect = metrics.boundingRect(text).translated(d->state.point);
  d->p->drawText(d->state.point, text);
  d->state.point.rx() += width;
  d->state.hasPoint = false;
  d->damage(brect);
}

static QTextDocument*
makeTextDocument(QString text, QRectF &rect, const QtLuaPainter::State &state, int flags)
{
  QTextDocument *doc = new QTextDocument;
  QTextFrameFormat fmt = doc->rootFrame()->frameFormat();
  fmt.setMargin(0);
  doc->rootFrame()->setFrameFormat(fmt);
  doc->setTextWidth(rect.width());
  doc->setDefaultFont(state.font);
  doc->setDefaultStyleSheet(state.styleSheet);
  QString div = "<div style=\"color: " + state.brush.color().name() + "\"";
  if (flags & Qt::AlignRight)
    div += "align=right";
  else if (flags & Qt::AlignHCenter)
    div += "align=center";
  else if (flags & Qt::AlignJustify)
    div += "align=justify";
  doc->setHtml(div + ">" + text + "</div>");
  QRectF lr = QRectF(QPointF(0,0), doc->documentLayout()->documentSize());
  qreal dy = rect.top();
  if (flags & Qt::AlignBottom)
    dy = rect.bottom() - lr.bottom();
  else if (flags & Qt::AlignVCenter)
    dy = rect.center().y() - lr.center().y();
  rect = lr.translated(rect.left(),dy).intersected(rect);
  return doc;
}



void 
QtLuaPainter::show(QString text, qreal x, qreal y, qreal w, qreal h, int flags)
{
  Locker lock(this);
  QRectF rect(x,y,w,h);
  if ((flags & TextRich) == TextRich)
    {
      // rich text
      QTextDocument *doc = makeTextDocument(text, rect, d->state, flags);
      d->p->save();
      if (d->state.clip.isEmpty() || d->state.clip.intersects(rect))
        {
          d->p->setClipRect(rect, Qt::IntersectClip);
          d->p->translate(rect.x(),rect.y());
          doc->drawContents(d->p);
        }
      d->p->restore();
      d->damage(rect);
      delete doc;
    }
  else
    {
      // normal text
      QRectF brect = d->p->boundingRect(rect, flags, text);
      d->p->drawText(rect, flags, text);
      d->damage(brect.intersected(rect));
    }
}


qreal
QtLuaPainter::stringwidth(QString text, qreal *pdx, qreal *pdy)
{
  QFontMetricsF metrics(d->state.font);
  qreal width = metrics.width(text);
  qreal height = 0; // qt limitation?
  if (pdx) 
    *pdx = width;
  if (pdy) 
    *pdy = height;
  return width;
}


QRectF 
QtLuaPainter::stringrect(QString text)
{
  QFontMetricsF metrics(d->state.font);
  return metrics.boundingRect(text).translated(d->state.point);
}


QRectF 
QtLuaPainter::stringrect(QString text, qreal x, qreal y, qreal w, qreal h, int flags)
{
  Locker lock(this);
  QRectF rect(x,y,w,h);
  if ((flags & TextRich) == TextRich)
    {
      // rich text
      QTextDocument *doc = makeTextDocument(text, rect, d->state, flags);
      delete doc;
      return rect;
    }
  else
    {
      // normal text
      return d->p->boundingRect(rect, flags, text);
    }
}


void 
QtLuaPainter::image(QRectF drect, QPixmap p, QRectF srect)
{
  Locker lock(this);
  d->p->drawPixmap(drect, p, srect);
  d->damage(drect);
}


void 
QtLuaPainter::image(QRectF drect, QImage i, QRectF srect)
{
  Locker lock(this);
  d->p->drawImage(drect, i, srect);
  d->damage(drect);
}


void 
QtLuaPainter::image(QRectF drect, QtLuaPainter *p, QRectF srect)
{
  Locker lock(this);
  d->p->drawImage(drect, p->image(), srect);
  d->damage(drect);
}




// ========================================
// MOC


#include "qtluapainter.moc"





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*" "qreal")
   End:
   ------------------------------------------------------------- */
