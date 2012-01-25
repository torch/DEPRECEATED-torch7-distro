// -*- C++ -*-

#ifndef QTLUAPAINTER_H
#define QTLUAPAINTER_H

#include "lua.h"
#include "lauxlib.h"
#include "qtluaengine.h"
#include "qtluautils.h"

#include "qtwidget.h"

#include <QBrush>
#include <QByteArray>
#include <QFlags>
#include <QMetaType>
#include <QImage>
#include <QObject>
#include <QPaintDevice>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPoint>
#include <QPrinter>
#include <QRegion>
#include <QTransform>
#include <QVariant>
#include <QWidget>

class QEvent;
class QCloseEvent;
class QFocusEvent;
class QPaintEvent;
class QResizeEvent;
class QKeyEvent;
class QMouseEvent;
class QMainWindow;
class QtLuaPrinter;

Q_DECLARE_METATYPE(QGradient)
Q_DECLARE_METATYPE(QPainterPath)
Q_DECLARE_METATYPE(QPolygon)
Q_DECLARE_METATYPE(QPolygonF)
Q_DECLARE_METATYPE(QPainter*)
Q_DECLARE_METATYPE(QPrinter*)
Q_DECLARE_METATYPE(QPaintDevice*)

class QTWIDGET_API QtLuaPainter : public QObject
{
  Q_OBJECT
  Q_PROPERTY(QPen pen READ currentpen WRITE setpen)
  Q_PROPERTY(QBrush brush READ currentbrush WRITE setbrush)
  Q_PROPERTY(QPointF point READ currentpoint WRITE setpoint)
  Q_PROPERTY(QPainterPath path READ currentpath WRITE setpath)
  Q_PROPERTY(QPainterPath clippath READ currentclip WRITE setclip)
  Q_PROPERTY(QFont font READ currentfont WRITE setfont)
  Q_PROPERTY(QTransform matrix READ currentmatrix WRITE setmatrix)
    // special
  Q_PROPERTY(QBrush background READ currentbackground WRITE setbackground)
  Q_PROPERTY(CompositionMode compositionmode READ currentmode WRITE setmode)
  Q_PROPERTY(RenderHints renderhints READ currenthints WRITE sethints)
  Q_PROPERTY(AngleUnit angleUnit READ currentangleunit WRITE setangleunit)
  Q_PROPERTY(QString styleSheet READ currentstylesheet WRITE setstylesheet)
  Q_PROPERTY(int width READ width)
  Q_PROPERTY(int height READ height)
  Q_PROPERTY(int depth READ depth)
  Q_ENUMS(CompositionMode AngleUnit)
  Q_FLAGS(RenderHints TextFlags)

public:
  ~QtLuaPainter();
  QtLuaPainter();
  QtLuaPainter(QImage image);
  QtLuaPainter(QPixmap pixmap);
  QtLuaPainter(int w, int h, bool monochrome=false);
  QtLuaPainter(QString fileName, const char *format = 0);
  QtLuaPainter(QWidget *widget, bool buffered=true);
  QtLuaPainter(QObject *object);

  Q_INVOKABLE QImage image() const;
  Q_INVOKABLE QPixmap pixmap() const;
  Q_INVOKABLE QWidget *widget() const;  
  Q_INVOKABLE QObject *object() const;
  Q_INVOKABLE QPaintDevice *device() const;
  Q_INVOKABLE QPrinter *printer() const;
  Q_INVOKABLE QPainter *painter() const;
  Q_INVOKABLE QRect rect() const;
  Q_INVOKABLE QSize size() const;
  Q_INVOKABLE void close();
  int width() const { return size().width(); }
  int height() const { return size().height(); }
  int depth() const;

  enum AngleUnit { Degrees, Radians };

  // copy qpainter enums for moc!
  enum CompositionMode {
    SourceOver = QPainter::CompositionMode_SourceOver,
    DestinationOver = QPainter::CompositionMode_DestinationOver,
    Clear = QPainter::CompositionMode_Clear,
    Source = QPainter::CompositionMode_Source,
    Destination = QPainter::CompositionMode_Destination,
    SourceIn = QPainter::CompositionMode_SourceIn,
    DestinationIn = QPainter::CompositionMode_DestinationIn,
    SourceOut = QPainter::CompositionMode_SourceOut,
    DestinationOut = QPainter::CompositionMode_DestinationOut,
    SourceAtop = QPainter::CompositionMode_SourceAtop,
    DestinationAtop = QPainter::CompositionMode_DestinationAtop,
    Xor = QPainter::CompositionMode_Xor,
    Plus = QPainter::CompositionMode_Plus,
    Multiply = QPainter::CompositionMode_Multiply,
    Screen = QPainter::CompositionMode_Screen,
    Overlay = QPainter::CompositionMode_Overlay,
    Darken = QPainter::CompositionMode_Darken,
    Lighten = QPainter::CompositionMode_Lighten,
    ColorDodge = QPainter::CompositionMode_ColorDodge,
    ColorBurn = QPainter::CompositionMode_ColorBurn,
    HardLight = QPainter::CompositionMode_HardLight,
    SoftLight = QPainter::CompositionMode_SoftLight,
    Difference = QPainter::CompositionMode_Difference,
    Exclusion = QPainter::CompositionMode_Exclusion
  };
  enum RenderHint {
    Antialiasing = QPainter::Antialiasing,
    TextAntialiasing = QPainter::TextAntialiasing,
    SmoothPixmapTransform = QPainter::SmoothPixmapTransform,
    HighQualityAntialiasing = QPainter::HighQualityAntialiasing,
  };
  enum TextFlag {
    AlignLeft = Qt::AlignLeft,
    AlignRight = Qt::AlignRight,
    AlignHCenter = Qt::AlignHCenter,
    AlignJustify = Qt::AlignJustify,
    AlignTop = Qt::AlignTop,
    AlignBottom = Qt::AlignBottom,
    AlignVCenter = Qt::AlignVCenter,
    AlignCenter = Qt::AlignCenter,
    TextSingleLine  =Qt::TextSingleLine,
    TextExpandTabs = Qt::TextExpandTabs,
    TextShowMnemonic = Qt::TextShowMnemonic,
    TextWordWrap = Qt::TextWordWrap,
    TextRich = (Qt::TextWordWrap|Qt::TextSingleLine), // magic
    RichText = TextRich                               // alias
  };
  Q_DECLARE_FLAGS(RenderHints,RenderHint);
  Q_DECLARE_FLAGS(TextFlags,TextFlag);

public slots:
  virtual void showpage();
  void refresh();
  
public:
  // buffering
  void gbegin();
  void gend(bool invalidate=false);
  // state
  QPen currentpen() const;
  QBrush currentbrush() const;
  QPointF currentpoint() const;
  QPainterPath currentpath() const;
  QPainterPath currentclip() const;
  QFont currentfont() const;
  QTransform currentmatrix() const;
  QBrush currentbackground() const;
  CompositionMode currentmode() const;
  RenderHints currenthints() const;
  AngleUnit currentangleunit() const;
  QString currentstylesheet() const;
  void setpen(QPen pen);
  void setbrush(QBrush brush);
  void setpoint(QPointF p);
  void setpath(QPainterPath p);
  void setclip(QPainterPath p);
  void setfont(QFont f);
  void setmatrix(QTransform m);
  void setbackground(QBrush brush);
  void setmode(CompositionMode m);
  void sethints(RenderHints h);
  void setangleunit(AngleUnit u);
  void setstylesheet(QString s);
  // postscript rendering
  void initclip();
  void initmatrix();
  void initgraphics();
  void scale(qreal x, qreal y);
  void rotate(qreal x);
  void translate(qreal x, qreal y);
  void concat(QTransform m);
  void gsave();
  void grestore();
  void newpath();
  void moveto(qreal x, qreal y);
  void lineto(qreal x, qreal y);
  void curveto(qreal x1, qreal y1, qreal x2, qreal y2, qreal x3, qreal y3);
  void arc(qreal x, qreal y, qreal r, qreal a1, qreal a2);
  void arcn(qreal x, qreal y, qreal r, qreal a1, qreal a2);
  void arcto(qreal x1, qreal y1, qreal x2, qreal y2, qreal r);
  void rmoveto(qreal x, qreal y);
  void rlineto(qreal x, qreal y);
  void rcurveto(qreal x1, qreal y1, qreal x2, qreal y2, qreal x3, qreal y3);
  void charpath(QString text);
  void closepath();
  void stroke(bool resetpath=true);
  void fill(bool resetpath=true);
  void eofill(bool resetpath=true);
  void clip(bool resetpath=false);
  void eoclip(bool resetpath=false);
  void show(QString text);
  qreal stringwidth(QString text, qreal *pdx=0, qreal *pdy=0);
  // additional useful functions
  void rectangle(qreal x, qreal y, qreal w, qreal h); // non ps
  void image(QRectF drect, QImage i, QRectF srect);
  void image(QRectF drect, QPixmap p, QRectF srect);
  void image(QRectF drect, QtLuaPainter *p, QRectF srect);
  void show(QString text, qreal x, qreal y, qreal w, qreal h, int flags=0);
  QRectF stringrect(QString text);
  QRectF stringrect(QString text, qreal x, qreal y, qreal w, qreal h, int f=0);
  
public:
  struct Private;
  struct Locker;
  struct State;
protected:
  Private *d;
};



#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*" "qreal")
   End:
   ------------------------------------------------------------- */


