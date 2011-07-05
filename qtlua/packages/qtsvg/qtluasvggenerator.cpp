/* -*- C++ -*- */


#include "qtluasvggenerator.h"

#include <QBuffer>

struct QtLuaSvgGenerator::Private
{
  QString description;
  QSize size;
  QString title;
  int resolution;
  QBuffer *buffer;
};


QtLuaSvgGenerator::~QtLuaSvgGenerator()
{
  emit closing(this);
  delete d;
}


QtLuaSvgGenerator::QtLuaSvgGenerator(QObject *parent)
  : QObject(parent), d(new Private)
{
  d->buffer = new QBuffer(this);
  QSvgGenerator::setOutputDevice(d->buffer);
  d->resolution = QSvgGenerator::resolution();
  d->size = QSvgGenerator::size();
}


QtLuaSvgGenerator::QtLuaSvgGenerator(QString fileName, QObject *parent)
  : QObject(parent), d(new Private)
{
  d->buffer = 0;
  QSvgGenerator::setFileName(fileName);
  d->resolution = QSvgGenerator::resolution();
  d->size = QSvgGenerator::size();
}


QString 
QtLuaSvgGenerator::description() const
{
  return d->description;
}


QSize 
QtLuaSvgGenerator::size() const
{
  return d->size;
}


QString 
QtLuaSvgGenerator::title() const
{
  return d->title;
}


int 
QtLuaSvgGenerator::resolution() const
{
  return d->resolution;
}


QByteArray
QtLuaSvgGenerator::data()
{
  if (d->buffer)
    return d->buffer->data();
  return QByteArray();
}


void
QtLuaSvgGenerator::setDescription(QString s)
{
  d->description = s;
#if QT_VERSION >= 0x40500
  QSvgGenerator::setDescription(s);
#endif
}


void
QtLuaSvgGenerator::setSize(QSize s)
{
  d->size = s;
  QSvgGenerator::setSize(s);
#if QT_VERSION >= 0x40500
  QSvgGenerator::setViewBox(QRect(QPoint(0,0), s));
#endif
}


void
QtLuaSvgGenerator::setTitle(QString s)
{
  d->title = s;
#if QT_VERSION >= 0x40500
  QSvgGenerator::setTitle(s);
#endif
}


void
QtLuaSvgGenerator::setResolution(int r)
{
  d->resolution = r;
  QSvgGenerator::setResolution(r);
}




/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
