// -*- C++ -*-

#ifndef QTHUMBWHEEL_H
#define QTHUMBWHEEL_H

#include <QtGlobal>
#include <QAbstractSlider>
#include <QSize>
#include <QWidget>

class QMouseEvent;
class QPaintEvent;
class QResizeEvent;

// Reimplemented  the thumbwheel widget from qt solutions from scratch.
// http://doc.trolltech.com/solutions/4/qtthumbwheel/qtthumbwheel.html.
// Note: limitedDrag does nothing.

class QThumbWheel : public QAbstractSlider
{
  Q_OBJECT
  Q_PROPERTY(int cogCount 
             READ cogCount WRITE setCogCount)
  Q_PROPERTY(double transmissionRatio 
             READ transmissionRatio WRITE setTransmissionRatio)
  Q_PROPERTY(bool wrapsAround
             READ wrapsAround WRITE setWrapsAround)
public:
  QThumbWheel(QWidget *parent = 0);
  QThumbWheel(int mi, int ma, int st, int v, Qt::Orientation o, QWidget *p=0);
  ~QThumbWheel();
  int  cogCount();
  double transmissionRatio();
  bool wrapsAround();
public slots:
  void setCogCount(int);
  void setTransmissionRatio(double);
  void setWrapsAround(bool);

protected:
  QSize sizeHint() const;
  QSize minimumSizeHint() const;
  void resizeEvent(QResizeEvent *);
  void paintEvent(QPaintEvent*);
  void mousePressEvent(QMouseEvent*);
  void mouseReleaseEvent(QMouseEvent*);
  void mouseMoveEvent(QMouseEvent*);

private:
  struct Private;
  Private *d;
};


#endif




/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ( "\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" )
   End:
   ------------------------------------------------------------- */

