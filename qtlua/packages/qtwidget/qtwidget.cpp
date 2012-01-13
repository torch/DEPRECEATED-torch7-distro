/* -*- C++ -*- */


#include "qtwidget.h"

#include "qtlualistener.h"
#include "qtluapainter.h"
#include "qtluaprinter.h"

#ifdef LUA_NOT_CXX
#include "lua.hpp"
#else
#include "lauxlib.h"
#include "lualib.h"
#endif

#include <QAction>
#include <QApplication>
#include <QBrush>
#include <QColor>
#include <QDebug>
#include <QFile>
#include <QFont>
#include <QGradient>
#include <QImage>
#include <QImageWriter>
#include <QMenu>
#include <QMenuBar>
#include <QMetaEnum>
#include <QMetaObject>
#include <QMetaType>
#include <QObject>
#include <QPen>
#include <QTransform>
#include <QVariant>
#include <QVector>


// ========================================
// HELPERS

  
static QMetaEnum 
f_enumerator(const char *s, const QMetaObject *mo)
{
  int index = mo->indexOfEnumerator(s);
  if (mo >= 0)
    return mo->enumerator(index);
  return QMetaEnum();
}


#ifndef Q_MOC_RUN
static QMetaEnum 
f_enumerator(const char *s)
{
  struct QFakeObject : public QObject {
    static const QMetaObject* qt() { return &staticQtMetaObject; } };
  return f_enumerator(s, QFakeObject::qt());
}
#endif


static void
f_checktype(lua_State *L, int index, const char *name, int type)
{ 
  if (index)
    lua_getfield(L,index,name); 
  int t = lua_type(L, -1);
  if (t != type)
    luaL_error(L, "%s expected in field '%s', got %s",
               lua_typename(L, type), name, lua_typename(L, t));
}

static bool
f_opttype(lua_State *L, int index, const char *name, int type)
{ 
  lua_getfield(L,index,name); 
  if (lua_isnoneornil(L, -1))
    return false;
  f_checktype(L, 0, name, type);
  return true;
}

static void
f_pushflag(lua_State *L, int value, const QMetaEnum &me)
{
  QByteArray b;
  if (me.isValid() && me.isFlag())
    b = me.valueToKeys(value);
  else if (me.isValid())
    b = me.valueToKey(value);
  if (b.size() > 0)
    lua_pushstring(L, b.constData());
  else
    lua_pushinteger(L, value);
}

static void
f_checkflag(lua_State *L, int index, const char *name, const QMetaEnum &me)
{ 
  if (index)
    lua_getfield(L,index,name); 
  if (me.isValid() && lua_isstring(L, -1))
    {
      const char *s = lua_tostring(L, -1);
      int v = (me.isFlag()) ? me.keysToValue(s) : me.keyToValue(s);
      if (v == -1)
        luaL_error(L, "value '%s' from field '%s' illegal for type %s::%s", 
                   s, name, me.scope(), me.name() );
      lua_pushinteger(L, v);
      lua_replace(L, -2);
    }
  if (! lua_isnumber(L, -1))
    {
      if (! me.isValid())
        luaL_error(L, "integer expected in field '%s'", name);
      luaL_error(L, "%s::%s or integer expected in field '%s'",
                 me.scope(), me.name(), name);
    }
}

static bool
f_optflag(lua_State *L, int index, const char *name, const QMetaEnum &me)
{ 
  lua_getfield(L,index,name); 
  if (lua_isnoneornil(L, -1))
    return false;
  f_checkflag(L, 0, name, me);
  return true;
}

static void
f_checkvar(lua_State *L, int index, const char *name, int tid)
{ 
  if (index)
    lua_getfield(L,index,name); 
  QVariant v = luaQ_toqvariant(L, -1, tid);
  if (v.userType() != tid)
    luaL_error(L, "qt.%s expected in field '%s'", QMetaType::typeName(tid));
}

static bool
f_optvar(lua_State *L, int index, const char *name, int tid)
{ 
  lua_getfield(L,index,name); 
  if (lua_isnoneornil(L, -1))
    return false;
  f_checkvar(L, 0, name, tid);
  return true;
}

template<typename T> static T
luaQE_checkqvariant(lua_State *L, int index, T* = 0)
{
  int type = qMetaTypeId<T>();
  QVariant v = luaQ_toqvariant(L, index, type);
  if (v.userType() != type)
    {
      luaQ_pushmeta(L, type);
      if (lua_istable(L, -1))
        {
          lua_getfield(L, -1, "__metatable");
          if (lua_istable(L, -1))
            {
              lua_getfield(L, -1, "new");
              if (lua_isfunction(L, -1))
                {
                  lua_pushvalue(L, index);
                  lua_call(L, 1, 1);
                  v = luaQ_toqvariant(L, -1, type);
                }
              lua_pop(L, 1);
            }
          lua_pop(L, 1);
        }
      lua_pop(L, 1);
    }
  if (v.userType() != type)
    luaL_typerror(L, index, QMetaType::typeName(type));
  return qVariantValue<T>(v);
}



// ========================================
// QTLUAPAINTER


static int
qtluapainter_new(lua_State *L)
{
  QtLuaPainter *p = 0;
  QObject *o = luaQ_toqobject(L, 1);
  QVariant v = luaQ_toqvariant(L, 1);
  if (v.userType() == QMetaType::QPixmap)
    {
      p = new QtLuaPainter(qVariantValue<QPixmap>(v));
    }
  else if (v.userType() == QMetaType::QImage)
    {
      p = new QtLuaPainter(qVariantValue<QImage>(v));
    }
  else if (qobject_cast<QWidget*>(o))
    {
      QWidget *w = qobject_cast<QWidget*>(o);
      bool buffered = true;
      if (! lua_isnone(L, 2))
        {
          luaL_checktype(L, 2, LUA_TBOOLEAN);
          buffered = lua_toboolean(L, 2);
        }
      p = new QtLuaPainter(w, buffered);
    }
  else if (o != 0)
    {
      p = new QtLuaPainter(o);
    }
  else if (lua_type(L, 1) == LUA_TSTRING)
    {
      const char *f = luaL_checkstring(L, 1);
      const char *format = luaL_optstring(L, 2, 0);
      p = new QtLuaPainter(QString::fromLocal8Bit(f), format);
      if (p->image().isNull())
        luaL_error(L,"cannot load image from file '%s'", f);
    }
  else if (lua_isuserdata(L, 1))
    {
      QFile f;
      void *udata = luaL_checkudata(L, 1, LUA_FILEHANDLE);
      const char *format = luaL_optstring(L, 2, 0);
      if (! f.open(*(FILE**)udata, QIODevice::ReadOnly))
        luaL_error(L,"cannot use stream for reading (%s)", 
                   f.errorString().toLocal8Bit().constData() );
      QImage img;
      if(! img.load(&f, format))
        luaL_error(L,"cannot load image from file");
      
    }
  else
    {
      int w = luaL_checkinteger(L, 1);
      int h = luaL_checkinteger(L, 2);
      bool monochrome = lua_toboolean(L, 3);
      if (! lua_isnone(L, 3))
        luaL_checktype(L, 3, LUA_TBOOLEAN);
      p = new QtLuaPainter(w, h, monochrome);
    }
  luaQ_pushqt(L, p, true);
  return 1;
}


#define qtluapainter_v(t) \
static int qtluapainter_ ## t (lua_State *L) { \
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);\
  p->t(); \
  return 0; }

qtluapainter_v(gbegin)
qtluapainter_v(refresh)
qtluapainter_v(initgraphics)
qtluapainter_v(initclip)
qtluapainter_v(initmatrix)
qtluapainter_v(gsave)
qtluapainter_v(grestore)
qtluapainter_v(newpath)
qtluapainter_v(closepath)
qtluapainter_v(showpage)


#define qtluapainter_V(t,V) \
static int qtluapainter_ ## t (lua_State *L) { \
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);\
  luaQ_pushqt(L, qVariantFromValue<V>(p->t())); \
  return 1; }

qtluapainter_V(rect, QRect)
qtluapainter_V(size, QSize)
qtluapainter_V(currentpen, QPen)
qtluapainter_V(currentbrush, QBrush)
qtluapainter_V(currentpoint, QPointF)
qtluapainter_V(currentpath, QPainterPath)
qtluapainter_V(currentclip, QPainterPath)
qtluapainter_V(currentfont, QFont)
qtluapainter_V(currentmatrix, QTransform)
qtluapainter_V(currentbackground, QBrush)
qtluapainter_V(currentstylesheet, QString)


static int 
qtluapainter_currentmode(lua_State *L)
{
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);
  const QMetaObject *mo = &QtLuaPainter::staticMetaObject;
  QMetaEnum e = f_enumerator("CompositionMode", mo);
  QtLuaPainter::CompositionMode s = p->currentmode();
  lua_pushstring(L, e.valueToKey((int)(s)));
  return 1; 
}

               
static int 
qtluapainter_currenthints(lua_State *L)
{
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);
  const QMetaObject *mo = &QtLuaPainter::staticMetaObject;
  QMetaEnum e = f_enumerator("RenderHints", mo);
  QtLuaPainter::RenderHints s = p->currenthints();
  lua_pushstring(L, e.valueToKeys((int)(s)).constData());
  return 1; 
}


static int 
qtluapainter_currentangleunit(lua_State *L)
{
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);
  const QMetaObject *mo = &QtLuaPainter::staticMetaObject;
  QMetaEnum e = f_enumerator("AngleUnit", mo);
  QtLuaPainter::AngleUnit s = p->currentangleunit();
  lua_pushstring(L, e.valueToKeys((int)(s)).constData());
  return 1; 
}


#define qtluapainter_vV(t,V) \
static int qtluapainter_ ## t (lua_State *L) { \
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);\
  V v = luaQE_checkqvariant<V>(L, 2); \
  p->t(v); \
  return 0; }

qtluapainter_vV(setpoint, QPointF)
qtluapainter_vV(setpath, QPainterPath)
qtluapainter_vV(setclip, QPainterPath)
qtluapainter_vV(setmatrix, QTransform)
qtluapainter_vV(concat, QTransform)
qtluapainter_vV(charpath, QString)
qtluapainter_vV(setstylesheet, QString)
qtluapainter_vV(setpen, QPen)
qtluapainter_vV(setbrush, QBrush)
qtluapainter_vV(setfont, QFont)
qtluapainter_vV(setbackground, QBrush)


static int 
qtluapainter_setmode(lua_State *L)
{
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);
  const char *s = luaL_checkstring(L, 2);
  const QMetaObject *mo = &QtLuaPainter::staticMetaObject;
  QMetaEnum e = f_enumerator("CompositionMode", mo);
  int x = e.keyToValue(s);
  luaL_argcheck(L, x>=0, 2, "unrecognized composition mode");
  p->setmode((QtLuaPainter::CompositionMode)x);
  return 0;
}


static int 
qtluapainter_sethints(lua_State *L)
{
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);
  const char *s = luaL_checkstring(L, 2);
  const QMetaObject *mo = &QtLuaPainter::staticMetaObject;
  QMetaEnum e = f_enumerator("RenderHints", mo);
  int x = e.keysToValue(s);
  luaL_argcheck(L, x>=0, 2, "unrecognized render hints");
  p->sethints((QtLuaPainter::RenderHints)x);
  return 0;
}

static int 
qtluapainter_setangleunit(lua_State *L)
{
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);
  const char *s = luaL_checkstring(L, 2);
  const QMetaObject *mo = &QtLuaPainter::staticMetaObject;
  QMetaEnum e = f_enumerator("AngleUnit", mo);
  int x = e.keysToValue(s);
  luaL_argcheck(L, x>=0, 2, "unrecognized render hints");
  p->setangleunit((QtLuaPainter::AngleUnit)x);
  return 0;
}



#define qtluapainter_vr(t) \
static int qtluapainter_ ## t (lua_State *L) { \
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);\
  qreal r = luaL_checknumber(L, 2); \
  p->t(r); \
  return 0; }

qtluapainter_vr(rotate)


#define qtluapainter_vrr(t) \
static int qtluapainter_ ## t (lua_State *L) { \
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);\
  qreal x = luaL_checknumber(L, 2); \
  qreal y = luaL_checknumber(L, 3); \
  p->t(x,y); \
  return 0; }

qtluapainter_vrr(scale)
qtluapainter_vrr(translate)
qtluapainter_vrr(moveto)
qtluapainter_vrr(lineto)
qtluapainter_vrr(rmoveto)
qtluapainter_vrr(rlineto)


#define qtluapainter_vrrrrr(t) \
static int qtluapainter_ ## t (lua_State *L) { \
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);\
  qreal x1 = luaL_checknumber(L, 2); \
  qreal y1 = luaL_checknumber(L, 3); \
  qreal x2 = luaL_checknumber(L, 4); \
  qreal y2 = luaL_checknumber(L, 5); \
  qreal r = luaL_checknumber(L, 6); \
  p->t(x1,y1,x2,y2,r); \
  return 0; }

qtluapainter_vrrrrr(arc)
qtluapainter_vrrrrr(arcn)
qtluapainter_vrrrrr(arcto)


#define qtluapainter_vrrrrrr(t) \
static int qtluapainter_ ## t (lua_State *L) { \
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);\
  qreal x1 = luaL_checknumber(L, 2); \
  qreal y1 = luaL_checknumber(L, 3); \
  qreal x2 = luaL_checknumber(L, 4); \
  qreal y2 = luaL_checknumber(L, 5); \
  qreal x3 = luaL_checknumber(L, 6); \
  qreal y3 = luaL_checknumber(L, 7); \
  p->t(x1,y1,x2,y2,x3,y3); \
  return 0; }

qtluapainter_vrrrrrr(curveto)
qtluapainter_vrrrrrr(rcurveto)


#define qtluapainter_vb(t) \
static int qtluapainter_ ## t (lua_State *L) { \
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);\
  if (lua_isnone(L, 2)) { p->t(); return 0; } \
  luaL_checktype(L, 2, LUA_TBOOLEAN); \
  p->t(lua_toboolean(L, 2)); \
  return 0; }

qtluapainter_vb(gend)
qtluapainter_vb(stroke)
qtluapainter_vb(fill)
qtluapainter_vb(eofill)
qtluapainter_vb(clip)
qtluapainter_vb(eoclip)


static int qtluapainter_show(lua_State *L)
{
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);
  QString s =luaQ_checkqvariant<QString>(L, 2);
  if (lua_gettop(L) == 2)
    {
      p->show(s);
    }
  else
    {
      qreal x = luaL_checknumber(L, 3);
      qreal y = luaL_checknumber(L, 4); 
      qreal w = luaL_checknumber(L, 5); 
      qreal h = luaL_checknumber(L, 6);
      const char *sf = luaL_optstring(L, 7, "AlignLeft");
      const QMetaObject *mo = &QtLuaPainter::staticMetaObject;
      QMetaEnum e = f_enumerator("TextFlags", mo);
      int f = e.keysToValue(sf);
      luaL_argcheck(L, f>=0, 7, "unrecognized flag");
      p->show(s,x,y,w,h,f);
    }
  return 0;
}


static int qtluapainter_stringwidth(lua_State *L)
{
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);
  QString s =luaQ_checkqvariant<QString>(L, 2);
  qreal dx, dy;
  p->stringwidth(s, &dx, &dy);
  lua_pushnumber(L, dx);
  lua_pushnumber(L, dy);
  return 2;
}


static int qtluapainter_stringrect(lua_State *L)
{
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);
  QString s =luaQ_checkqvariant<QString>(L, 2);
  if (lua_gettop(L) == 2)
    {
      luaQ_pushqt(L, qVariantFromValue(p->stringrect(s)));
    }
  else
    {
      qreal x = luaL_checknumber(L, 3);
      qreal y = luaL_checknumber(L, 4); 
      qreal w = luaL_checknumber(L, 5); 
      qreal h = luaL_checknumber(L, 6);
      const char *sf = luaL_optstring(L, 7, "");
      const QMetaObject *mo = &QtLuaPainter::staticMetaObject;
      QMetaEnum e = f_enumerator("TextFlags", mo);
      int f = e.keysToValue(sf);
      luaL_argcheck(L, f>=0, 7, "unrecognized flag");
      luaQ_pushqt(L, qVariantFromValue(p->stringrect(s,x,y,w,h,f)));
    }
  return 1;
}


static int qtluapainter_rectangle(lua_State *L)
{
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);
  qreal x =luaL_checknumber(L, 2);
  qreal y =luaL_checknumber(L, 3);
  qreal w =luaL_checknumber(L, 4);
  qreal h =luaL_checknumber(L, 5);
  p->rectangle(x,y,w,h);
  return 0;
}


static int qtluapainter_image(lua_State *L)
{
  QtLuaPainter *p = luaQ_checkqobject<QtLuaPainter>(L, 1);
  if (lua_gettop(L) == 1)
    {
      luaQ_pushqt(L, QVariant(p->image()));
      return 1;
    }
  qreal x, y, w=-1, h=-1;
  qreal sx=0, sy=0, sw=-1, sh=-1;
  int k = 2;
  x = luaL_checknumber(L, k++);
  y = luaL_checknumber(L, k++);
  if (lua_isnumber(L, k)) {
    w = luaL_checknumber(L, k++);
    h = luaL_checknumber(L, k++);
  }
  QtLuaPainter *o = qobject_cast<QtLuaPainter*>(luaQ_toqobject(L, k));
  QVariant v = luaQ_toqvariant(L, k);
  if (lua_istable(L, k)) {
    lua_getfield(L, k, "image");
    if (lua_isfunction(L, -1)) {
      lua_pushvalue(L, k);
      lua_call(L, 1, 1);
      v = luaQ_toqvariant(L, -1);
    }
    lua_pop(L, 1);
  }
  if (v.userType() == QMetaType::QImage) {
    QImage q = qVariantValue<QImage>(v);
    sw = q.width(); 
    sh = q.height(); 
  } else if (v.userType() == QMetaType::QPixmap) {
    QPixmap q = qVariantValue<QPixmap>(v);
    sw = q.width(); 
    sh = q.height();
  } else if (o) {
    sw = o->width();
    sh = o->height();
  } else
    luaL_typerror(L, k, "QPixmap or QImage");
  k += 1;
  if (lua_isnumber(L, k)) {
    sx = luaL_checknumber(L, k++);
    sy = luaL_checknumber(L, k++);
  }
  if (lua_isnumber(L, k)) {
    sw = luaL_checknumber(L, k++);
    sh = luaL_checknumber(L, k++);
  }
  if (w <= 0)
    w = sw;
  if (h <= 0)
    h = sh;
  QRectF dst(x,y,w,h);
  QRectF src(sx,sy,sw,sh);
  if (v.userType() == QMetaType::QPixmap)
    p->image(dst, qVariantValue<QPixmap>(v), src);
  else if (v.userType() == QMetaType::QImage)
    p->image(dst, qVariantValue<QImage>(v), src);
  else if (o)
    p->image(dst, o, src);    
  return 0;
}


struct luaL_Reg qtluapainter_lib[] = {
  {"rect", qtluapainter_rect},
  {"size", qtluapainter_size},
  {"gbegin", qtluapainter_gbegin},
  {"refresh", qtluapainter_refresh},
  {"gend", qtluapainter_gend},
  {"gsave", qtluapainter_gsave},
  {"grestore", qtluapainter_grestore},
  {"currentpen", qtluapainter_currentpen},
  {"currentbrush", qtluapainter_currentbrush},
  {"currentpoint", qtluapainter_currentpoint},
  {"currentpath", qtluapainter_currentpath},
  {"currentclip", qtluapainter_currentclip},
  {"currentfont", qtluapainter_currentfont},
  {"currentmatrix", qtluapainter_currentmatrix},
  {"currentbackground", qtluapainter_currentbackground},
  {"currentmode", qtluapainter_currentmode},
  {"currenthints", qtluapainter_currenthints},
  {"currentangleunit", qtluapainter_currentangleunit},
  {"currentstylesheet", qtluapainter_currentstylesheet},
  {"setpen", qtluapainter_setpen},
  {"setbrush", qtluapainter_setbrush},
  {"setfont", qtluapainter_setfont},
  {"setpoint", qtluapainter_setpoint},
  {"setpath", qtluapainter_setpath},
  {"setclip", qtluapainter_setclip},
  {"setmatrix", qtluapainter_setmatrix},
  {"setbackground", qtluapainter_setbackground},
  {"setmode", qtluapainter_setmode},
  {"sethints", qtluapainter_sethints},
  {"setangleunit", qtluapainter_setangleunit},
  {"setstylesheet", qtluapainter_setstylesheet},
  {"initclip", qtluapainter_initclip},
  {"initmatrix", qtluapainter_initmatrix},
  {"initgraphics", qtluapainter_initgraphics},
  {"scale", qtluapainter_scale},
  {"rotate", qtluapainter_rotate},
  {"translate", qtluapainter_translate},
  {"concat", qtluapainter_concat},
  {"newpath", qtluapainter_newpath},
  {"moveto", qtluapainter_moveto},
  {"lineto", qtluapainter_lineto},
  {"curveto", qtluapainter_curveto},
  {"arc", qtluapainter_arc},
  {"arcn", qtluapainter_arcn},
  {"arcto", qtluapainter_arcto},
  {"rmoveto", qtluapainter_rmoveto},
  {"rlineto", qtluapainter_rlineto},
  {"rcurveto", qtluapainter_rcurveto},
  {"closepath", qtluapainter_closepath},
  {"stroke", qtluapainter_stroke},
  {"fill", qtluapainter_fill},
  {"eofill", qtluapainter_eofill},
  {"clip", qtluapainter_clip},
  {"eoclip", qtluapainter_eoclip},
  {"image", qtluapainter_image},
  {"showpage", qtluapainter_showpage},
  {"rectangle", qtluapainter_rectangle},
  {0,0}
};


struct luaL_Reg qtluapainter_guilib[] = {
  // things that have to be done in the gui thread
  {"new", qtluapainter_new},
  {"charpath", qtluapainter_charpath},
  {"show", qtluapainter_show},
  {"stringwidth", qtluapainter_stringwidth},
  {"stringrect", qtluapainter_stringrect},
  {"stringrect", qtluapainter_stringrect},
  {0,0}
};


static int qtluapainter_hook(lua_State *L) 
{
  lua_getfield(L, -1, "__metatable");
  luaL_register(L, 0, qtluapainter_lib);
  // luaQ_register(L, qtluapainter_lib, QCoreApplication::instance());
  luaQ_register(L, qtluapainter_guilib, QCoreApplication::instance());
  return 0;
}





// ========================================
// LISTENER



static int qtlualistener_new(lua_State *L)
{
  QWidget *w = luaQ_checkqobject<QWidget>(L, 1);
  QtLuaListener *l = new QtLuaListener(w);
  l->setObjectName("listener");
  luaQ_pushqt(L, l);
  return 1;
}


static struct luaL_Reg qtlualistener_lib[] = {
  {"new", qtlualistener_new},
  {0,0}
};


static int qtlualistener_hook(lua_State *L) 
{
  lua_getfield(L, -1, "__metatable");
  luaQ_register(L, qtlualistener_lib, QCoreApplication::instance());
  return 0;
}



// ========================================
// QTLUAPRINTER


static int qtluaprinter_new(lua_State *L)
{
  static const char *modes[] = {"ScreenResolution","HighResolution",0};
  QPrinter::PrinterMode mode = QPrinter::ScreenResolution;
  if (luaL_checkoption(L, 1, "ScreenResolution", modes))
    mode = QPrinter::HighResolution;
  QtLuaPrinter *p = new QtLuaPrinter(mode);
  luaQ_pushqt(L, p, true);
  return 1;
}


static luaL_Reg qtluaprinter_lib[] = {
  {"new", qtluaprinter_new},
  {0,0}
};


static int qtluaprinter_hook(lua_State *L) 
{
  lua_getfield(L, -1, "__metatable");
  luaQ_register(L, qtluaprinter_lib, QCoreApplication::instance());
  return 0;
}



// ========================================
// REGISTER


#ifndef LUA_NOT_CXX
LUA_EXTERNC
#endif
QTWIDGET_API 
int luaopen_libqtwidget(lua_State *L)
{ 
  // load module 'qt'
  if (luaL_dostring(L, "require 'qt'"))
    lua_error(L);
  if (QApplication::type() == QApplication::Tty)
    printf("qtwidget window functions will not be usable (running with -nographics)\n");
  //luaL_error(L, "Graphics have been disabled (running with -nographics)");

  // register metatypes
  qRegisterMetaType<QPainter*>("QPainter*");
  qRegisterMetaType<QPrinter*>("QPrinter*");
  qRegisterMetaType<QPaintDevice*>("QPaintDevice*");

  // call hook for qobjects
#define HOOK_QOBJECT(T, t) \
     lua_pushcfunction(L, t ## _hook);\
     luaQ_pushmeta(L, &T::staticMetaObject);\
     lua_call(L, 1, 0)
  
  // call hook for qvariants
#define HOOK_QVARIANT(T, t) \
     lua_pushcfunction(L, t ## _hook);\
     luaQ_pushmeta(L, QMetaType::T);\
     lua_call(L, 1, 0)
  
  HOOK_QOBJECT(QtLuaPrinter, qtluaprinter);  
  HOOK_QOBJECT(QtLuaPainter, qtluapainter);  
  HOOK_QOBJECT(QtLuaListener, qtlualistener);  
  return 1;
}




/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


