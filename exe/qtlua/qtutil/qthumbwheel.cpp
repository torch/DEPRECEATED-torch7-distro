// -*- C++ -*-


#include <stdlib.h>
#include <math.h>

#include <QDebug>
#include <QAbstractSlider>
#include <QApplication>
#include <QEvent>
#include <QLinearGradient>
#include <QList>
#include <QMouseEvent>
#include <QObject>
#include <QPainter>
#include <QPaintEvent>
#include <QPoint>
#include <QPointF>
#include <QRect>
#include <QStyle>
#include <QStyleOption>
#include <QStyleOptionSlider>
#include <QSizePolicy>
#include <QWidget>

#include "qthumbwheel.h"

#ifndef M_PI
# define M_PI 3.14159265358979323846
#endif
#ifndef M_2PI
# define M_2PI (2*M_PI)
#endif


struct QThumbWheel::Private
{
  QThumbWheel *q;
  int cogCount;
  double transmissionRatio;
  bool wrapsAround;
  bool dragMode;
  int dragVal;
  QPoint dragPos;
  double cogAngle;
  double openAngle;
  QList<QPointF> cogs;
  QRect allRect;
  QRect tRect;
  QRect dRect;
  QRect uRect;
  double radius;
  double center;

  Private(QThumbWheel*);
  void computeCogs();
  int valueFromPos(QPoint opos, QPoint npos);

};


QThumbWheel::Private::Private(QThumbWheel *q)
  : q(q), cogCount(17), transmissionRatio(1.0), 
    wrapsAround(true), dragMode(false)
{
  openAngle = M_PI / 3;
  q->setFocusPolicy(Qt::WheelFocus);
  QSizePolicy sp(QSizePolicy::Preferred, QSizePolicy::Fixed);
  q->setSizePolicy(sp);
  q->setAttribute(Qt::WA_WState_OwnSizePolicy, false);
  q->setAttribute(Qt::WA_OpaquePaintEvent);
  computeCogs();
}


void 
QThumbWheel::Private::computeCogs()
{
  cogs.clear();
  cogAngle = M_2PI / cogCount;
  for (int i=0; i<cogCount; i++)
    {
      double angle = cogAngle * i;
      cogs << QPointF(cos(angle),sin(angle));
    }
}


int 
QThumbWheel::Private::valueFromPos(QPoint opos, QPoint npos)
{
  bool horiz = (q->orientation() == Qt::Horizontal);
  int maxi = q->maximum();
  int mini = q->minimum();
  int range = maxi - mini + 1;
  int dpos = (horiz) ? (npos.x()-opos.x()) : (opos.y()-npos.y());
  double delta = (dpos * range) / (radius * transmissionRatio * 2 * M_PI);
  int v = dragVal + (int)(delta);
  if (wrapsAround)
    {
      while (v < mini)
        v = v + (maxi - mini + 1);
      v = mini + ((v-mini) % range);
    }
  return qBound(mini,v,maxi);
}


QThumbWheel::QThumbWheel(QWidget *parent)
  : QAbstractSlider(parent),
    d(new Private(this))
{
  setOrientation(Qt::Vertical);
  setRange(-180,179);
  setPageStep(10);
  setCogCount(23);
  setValue(0);
  setWrapsAround(true);
}


QThumbWheel::QThumbWheel(int mi, int ma, int st, int v, 
                         Qt::Orientation o, QWidget *parent)
  : QAbstractSlider(parent),
    d(new Private(this))
{
  setOrientation(o);
  setRange(mi, ma);
  setPageStep(st);
  setValue(v);
  setWrapsAround(false);
}


QThumbWheel::~QThumbWheel()
{
  delete d;
}


int 
QThumbWheel::cogCount()
{
  return d->cogCount;
}


void 
QThumbWheel::setCogCount(int s)
{
  d->cogCount = s;
  d->computeCogs();
  update();
}


double 
QThumbWheel::transmissionRatio()
{
  return d->transmissionRatio;
}


void 
QThumbWheel::setTransmissionRatio(double s)
{
  d->transmissionRatio = s;
}


bool 
QThumbWheel::wrapsAround()
{
  return d->wrapsAround;
}


void 
QThumbWheel::setWrapsAround(bool s)
{
  d->wrapsAround = s;
}


QSize 
QThumbWheel::sizeHint() const
{
  ensurePolished();
  bool horiz = (orientation() == Qt::Horizontal);
  int sbExt = style()->pixelMetric(QStyle::PM_ScrollBarExtent, 0, this);
  int sbMin = style()->pixelMetric(QStyle::PM_ScrollBarSliderMin, 0, this);
  int v = sbExt+sbExt/4;
  int u = qMax(2*sbMin+2*sbExt, 6 * v);
  QSize size = (horiz) ? QSize(u,v) : QSize(v,u);
  size = style()->sizeFromContents(QStyle::CT_ScrollBar, 0, size, this);
  return size.expandedTo(QApplication::globalStrut());
}


QSize 
QThumbWheel::minimumSizeHint() const
{
  ensurePolished();
  bool horiz = (orientation() == Qt::Horizontal);
  int sbExt = style()->pixelMetric(QStyle::PM_ScrollBarExtent, 0, this);
  int sbMin = style()->pixelMetric(QStyle::PM_ScrollBarSliderMin, 0, this);
  int v = sbExt+sbExt/4;
  int u = 2*sbMin;
  QSize size = (horiz) ? QSize(u,v) : QSize(v,u);
  size = style()->sizeFromContents(QStyle::CT_ScrollBar, 0, size, this);
  return size.expandedTo(QApplication::globalStrut());
}


void 
QThumbWheel::resizeEvent(QResizeEvent *)
{
  QRect r = rect();
  d->tRect = d->uRect = d->dRect = r;
  if (orientation() == Qt::Horizontal)
    {
      int a = qMax(r.height()/2,16);
      d->uRect.setLeft(d->uRect.right()-a);
      d->dRect.setRight(d->dRect.left()+a);
      d->center = (r.left() + r.right())/2.0;
      d->radius = d->tRect.width() * 0.52;
    }
  else
    {
      int a = qMax(r.width()/2,16);
      d->uRect.setBottom(d->uRect.top()+a);
      d->dRect.setTop(d->dRect.bottom()-a);
      d->center = (r.top() + r.bottom())/2.0;
      d->radius = d->tRect.height() * 0.52;
    }
}


void 
QThumbWheel::paintEvent(QPaintEvent *e)
{
  bool horiz = (orientation() == Qt::Horizontal);
  QPainter painter(this);
  QPalette palette(this->palette());
  if (isEnabled())
    palette.setCurrentColorGroup(QPalette::Normal);
  else
    palette.setCurrentColorGroup(QPalette::Disabled);
  // main colors
  QRect cr = d->tRect;
  painter.setPen(Qt::NoPen);
  painter.setBrush(palette.window());
  painter.drawRect(rect());
  qDrawShadePanel(&painter, cr, palette, false, 1);
  // shadow
  if (horiz)
    cr.adjust(1,1,-1,-1);
  else
    cr.adjust(1,1,-1,-1);
  if (cr.isEmpty())
    return;
  QColor shadowColor = palette.color(QPalette::Shadow);
  qDrawPlainRect(&painter, cr, shadowColor, 1);
  // wheel
  if (horiz)
    cr.adjust(1,1,0,-1);
  else
    cr.adjust(1,1,-1,0);
  if (cr.isEmpty())
    return;
  QLinearGradient grad;
  if (horiz)
    grad = QLinearGradient(cr.topLeft(),cr.topRight());
  else
    grad = QLinearGradient(cr.topLeft(),cr.bottomLeft());
  grad.setColorAt(0, palette.color(QPalette::Mid));
  grad.setColorAt(0.35, palette.color(QPalette::Button));
  grad.setColorAt(1, palette.color(QPalette::Dark));
  QBrush gBrush(grad);
  painter.setBrush(gBrush);
  painter.drawRect(cr);
  // notches
  if (horiz)
    cr.adjust(0,2,0,-1);
  else
    cr.adjust(2,0,-1,0);
  if (cr.isEmpty() || d->radius <= 0)
    return;
  int maxi = maximum();
  int mini = minimum();
  int cval = 2 * sliderPosition() - maxi - mini;
  double angle = (M_PI * cval * d->transmissionRatio) / (maxi - mini + 1);
  double wAngle = qMin(d->cogAngle*0.1, 1.0/d->radius);
  double ca = cos(angle);
  double sa = sin(angle);
  double cb = cos(d->cogAngle-wAngle);
  double sb = sin(d->cogAngle-wAngle);
  painter.setClipRect(cr, Qt::IntersectClip);
  for (int i=0; i<d->cogCount; i++)
    {
      QPointF &cs = d->cogs[i];
      double c0 = ca * cs.x() - sa * cs.y();
      double s0 = ca * cs.y() + sa * cs.x(); 
      double c1 = c0 * cb - s0 * sb;
      double s1 = c0 * sb + s0 * cb;
      if (c0>=0 || c1 >= 0)
        {
          QRect nr = cr;
          double u0 = d->radius;
          double u1 = d->radius;
          if (c0 >= 0)
            u0 = d->radius * s0;
          if (c1 >= 0)
            u1 = d->radius * s1;
          if (horiz)
            {
              nr.setLeft((int)(d->center + u0));
              nr.setRight((int)(d->center + u1 - 1));
            } 
          else 
            {
              nr.setBottom((int)(d->center - u0 - 1));
              nr.setTop((int)(d->center - u1));
            }
          if (nr.isValid())
            qDrawShadePanel(&painter, nr, palette, true, 1, &gBrush);
        }
    }
  // adjust etches
  if (horiz)
    cr.adjust(0,cr.height()-1,0,1);
  else
    cr.adjust(cr.width()-1,0,1,0);
  painter.drawRect(cr);
}


void 
QThumbWheel::mousePressEvent(QMouseEvent *e)
{
  e->ignore();
  if (maximum() == minimum()) 
    return;
  if (e->button() != Qt::LeftButton || e->buttons() ^ e->button()) 
    return;
  e->accept();
  QAbstractSlider::SliderAction action = QAbstractSlider::SliderNoAction;
  if (d->uRect.contains(e->pos()))
    action = QAbstractSlider::SliderSingleStepAdd;
  else if (d->dRect.contains(e->pos()))
    action = QAbstractSlider::SliderSingleStepSub;
  else
    d->dragMode = true;
  if (action != QAbstractSlider::SliderNoAction)
    triggerAction(action);
  setRepeatAction(action,250,20);
  d->dragPos = e->pos();
  d->dragVal = value();
  setSliderDown(true);
}

void 
QThumbWheel::mouseReleaseEvent(QMouseEvent *e)
{
  e->ignore();
  if (!isSliderDown() || (e->buttons() & ~ e->button()))
    return;
  e->accept();
  if (d->dragMode)
    setValue(d->valueFromPos(d->dragPos, e->pos()));
  setRepeatAction(QAbstractSlider::SliderNoAction);
  d->dragMode = false;
  setSliderDown(false);
  update();
} 

void 
QThumbWheel::mouseMoveEvent(QMouseEvent *e)
{
  e->ignore();
  if (!isSliderDown() || !(e->buttons() & Qt::LeftButton))
    return;
  e->accept();
  if (!d->dragMode && d->tRect.contains(e->pos()))
    if (!d->dRect.contains(e->pos()) && !d->uRect.contains(e->pos()))
      {
        setRepeatAction(QAbstractSlider::SliderNoAction);
        d->dragMode = true;
      }
  if (d->dragMode)
    setSliderPosition(d->valueFromPos(d->dragPos, e->pos()));
}




#ifdef LLDEBUG
# include <QGridLayout>
# include <QLabel>

int 
main(int argc, char **argv)
{
  QApplication app(argc, argv);
  QWidget *w = new QWidget;
  w->setAttribute(Qt::WA_DeleteOnClose);
  QThumbWheel *w1 = new QThumbWheel(w);
  w1->setOrientation(Qt::Vertical);
  w1->setObjectName("vertical");
  w1->setTracking(false);
  QThumbWheel *w2 = new QThumbWheel(w);
  w2->setOrientation(Qt::Horizontal);
  w2->setObjectName("horizontal");
  w2->setTransmissionRatio(2);
  QLabel *l1p = new QLabel(w);
  QLabel *l1v = new QLabel(w);
  QLabel *l2p = new QLabel(w);
  QLabel *l2v = new QLabel(w);
  QWidget::connect(w1,SIGNAL(sliderMoved(int)), l1p, SLOT(setNum(int)));
  QWidget::connect(w1,SIGNAL(valueChanged(int)), l1v, SLOT(setNum(int)));
  QWidget::connect(w2,SIGNAL(sliderMoved(int)), l2p, SLOT(setNum(int)));
  QWidget::connect(w2,SIGNAL(valueChanged(int)), l2v, SLOT(setNum(int)));
  QGridLayout *l = new QGridLayout;
  l->addWidget(w1,0,5,5,1,Qt::AlignCenter);
  l->addWidget(w2,5,0,1,5,Qt::AlignCenter);
  l->addWidget(l1p,0,0);
  l->addWidget(l1v,0,1);
  l->addWidget(l2p,1,0);
  l->addWidget(l2v,1,1);
  l->setMargin(2);
  l->setSpacing(1);
  w->setLayout(l);
  w->resize(300,200);
  w->show();
  return app.exec();
}

#endif




/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ( "\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" )
   End:
   ------------------------------------------------------------- */
