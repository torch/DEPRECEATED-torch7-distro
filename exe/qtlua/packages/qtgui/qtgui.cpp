// -*- C++ -*-

#include "qtgui.h"
#include "qtluagui.h"

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>

#include "lauxlib.h"
#include "lualib.h"

#include <QAction>
#include <QApplication>
#include <QBitmap>
#include <QBrush>
#include <QColor>
#include <QColorDialog>
#include <QCursor>
#include <QDebug>
#include <QDialog>
#include <QFile>
#include <QFileDialog>
#include <QFont>
#include <QFontDialog>
#include <QFontInfo>
#include <QGradient>
#include <QIcon>
#include <QImage>
#include <QImageReader>
#include <QImageWriter>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMetaEnum>
#include <QMetaObject>
#include <QMetaType>
#include <QObject>
#include <QPainter>
#include <QPen>
#include <QStatusBar>
#include <QString>
#include <QTransform>
#include <QVariant>
#include <QVector>
#if HAVE_QTWEBKIT
#include <QWebView>
#endif

Q_DECLARE_METATYPE(QGradient)
Q_DECLARE_METATYPE(QPainterPath)
Q_DECLARE_METATYPE(QPolygon)
Q_DECLARE_METATYPE(QPolygonF)
Q_DECLARE_METATYPE(QPainter*)
Q_DECLARE_METATYPE(QPaintDevice*)



// ========================================
// HELPERS

  
static QMetaEnum 
f_enumerator(const char *s, const QMetaObject *mo)
{
  int index = mo->indexOfEnumerator(s);
  if (index >= 0)
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


#define fromtable_bool(n,get,set) \
  { f_checktype(L, -1, n, LUA_TBOOLEAN); \
    bool x = lua_toboolean(L, -1); set; \
    lua_pop(L, 1); }

#define fromtable_int(n,get,set) \
  { f_checktype(L, -1, n, LUA_TNUMBER); \
    lua_Integer x = lua_tointeger(L, -1); set; \
    lua_pop(L, 1); }

#define fromtable_flt(n,get,set) \
  { f_checktype(L, -1, n, LUA_TNUMBER); \
    lua_Number x = lua_tonumber(L, -1); set; \
    lua_pop(L, 1); }

#define fromtable_str(n,get,set) \
  { f_checktype(L, -1, n, LUA_TSTRING); \
    QString x = QString::fromLocal8Bit(lua_tostring(L, -1)); set; \
    lua_pop(L, 1); }

#define fromtable_optbool(n,get,set) \
  { if (f_opttype(L, -1, n, LUA_TBOOLEAN)) { \
      bool x = lua_toboolean(L, -1); set; } \
    lua_pop(L, 1); }

#define fromtable_optint(n,get,set) \
  { if (f_opttype(L, -1, n, LUA_TNUMBER)) { \
      lua_Integer x = lua_tointeger(L, -1); set; } \
    lua_pop(L, 1); }

#define fromtable_optflt(n,get,set) \
  { if (f_opttype(L, -1, n, LUA_TNUMBER)) { \
      lua_Number x = lua_tonumber(L, -1); set; } \
    lua_pop(L, 1); }

#define fromtable_optstr(n,get,set) \
  { if (f_opttype(L, -1, n, LUA_TSTRING)) { \
      QString x = QString::fromLocal8Bit(lua_tostring(L, -1)); set; } \
    lua_pop(L, 1); }

#define do_fromtable(T,t,do,declare,construct) \
static int t ## _fromtable(lua_State *L) \
{ \
  declare; \
  if (! lua_isnoneornil(L, 1)) { \
    luaL_checktype(L, 1, LUA_TTABLE); \
    do(fromtable_) } \
  luaQ_pushqt(L, QVariant(construct)); \
  return 1; \
}


#define totable_bool(n,get,set) \
  { bool x; get; \
    lua_pushboolean(L,x); \
    lua_setfield(L, -2, n); }

#define totable_int(n,get,set) \
  { lua_Integer x; get; \
    lua_pushinteger(L,x); \
    lua_setfield(L, -2, n); }

#define totable_flt(n,get,set) \
  { lua_Number x; get; \
    lua_pushnumber(L,x); \
    lua_setfield(L, -2, n); }

#define totable_str(n,get,set) \
  { QString x; get; \
    lua_pushstring(L,x.toLocal8Bit().constData()); \
    lua_setfield(L, -2, n); }

#define totable_optbool(n,get,set) \
  totable_bool(n,get,set)

#define totable_optint(n,get,set) \
  totable_int(n,get,set)

#define totable_optflt(n,get,set) \
  totable_flt(n,get,set)

#define totable_optstr(n,get,set) \
  totable_str(n,get,set)

#define do_totable(T,t,do) \
static int t ## _totable(lua_State *L) \
{ \
  T s = luaQ_checkqvariant<T>(L, 1); \
  lua_createtable(L, 0, 2); \
  do(totable_) \
  return 1; \
}

#define do_luareg(t) \
static struct luaL_Reg t ## _lib[] = {\
  {"totable", t ## _totable }, \
  {"new", t ## _fromtable }, \
  {0,0} \
}; \

#define do_hook(t) \
static int t ## _hook(lua_State *L) \
{ \
  lua_getfield(L, -1, "__metatable"); \
  luaL_register(L, 0, t ## _lib); \
  return 0; \
}

#define do_qhook(t) \
static int t ## _hook(lua_State *L) \
{ \
  lua_getfield(L, -1, "__metatable"); \
  luaQ_register(L, t ## _lib, QCoreApplication::instance()); \
  return 0; \
}




// ========================================
// QACTION, QTLUAACTION

static int
qaction_new(lua_State *L)
{
  if (lua_istable(L, 1))
    {
      QAction *action = new QAction(0);
      luaQ_pushqt(L, action, true);
      lua_pushnil(L);
      while (lua_next(L, 1)) 
        {                        // [val key action
          lua_pushvalue(L, -2);  // [key val key action
          lua_insert(L, -3);     // [val key key action
          if (lua_isfunction(L, -1)) {
            luaQ_connect(L, action, SIGNAL(triggered(bool)), -1);
            lua_pop(L, 2);
          } else
            lua_settable(L, -4); // [key action
        }
      return 1;
    }
  QWidget *parent = luaQ_optqobject<QWidget>(L, 1);
  QAction *action = new QAction(parent);
  luaQ_pushqt(L, action, !parent);
  return 1;
}


static int
qaction_menu(lua_State *L)
{
  QAction *a = luaQ_checkqobject<QAction>(L, 1);
  luaQ_pushqt(L, a->menu());
  return 1;
}


static int
qaction_setmenu(lua_State *L)
{
  QAction *a = luaQ_checkqobject<QAction>(L, 1);
  QMenu *m = luaQ_optqobject<QMenu>(L, 2);
  a->setMenu(m);
  return 0;
}


static struct luaL_Reg qaction_lib[] = {
  {"new", qaction_new},
  {"menu", qaction_menu},
  {"setMenu", qaction_setmenu},
  {0,0}
};

do_qhook(qaction)


static int
qtluaaction_new(lua_State *L)
{
  if (lua_istable(L, 1))
    {
      QtLuaAction *action = new QtLuaAction(luaQ_engine(L));
      luaQ_pushqt(L, action, true);
      lua_pushnil(L);
      while (lua_next(L, 1)) 
        {                        // [val key action
          lua_pushvalue(L, -2);  // [key val key action
          lua_insert(L, -3);     // [val key key action
          if (lua_isfunction(L, -1)) {
            luaQ_connect(L, action, SIGNAL(triggered(bool)), -1);
            lua_pop(L, 2);
          } else
            lua_settable(L, -4); // [key action
        }
      return 1;
    }
  QWidget *parent = luaQ_optqobject<QWidget>(L, 1);
  QtLuaAction *action = new QtLuaAction(luaQ_engine(L), parent);
  luaQ_pushqt(L, action, !parent);
  return 1;
}

static struct luaL_Reg qtluaaction_lib[] = {
  {"new", qtluaaction_new},
  {0,0}
};

do_qhook(qtluaaction)




// ========================================
// QAPPLICATION


static int
qapplication_keyboard_modifiers(lua_State *L)
{
  const QMetaEnum me = f_enumerator("KeyboardModifiers");
  lua_pushstring(L, me.valueToKeys(QApplication::keyboardModifiers()));
  return 1;
}

static int
qapplication_mouse_buttons(lua_State *L)
{
  const QMetaEnum me = f_enumerator("MouseButtons");
  lua_pushstring(L, me.valueToKeys(QApplication::mouseButtons()));
  return 1;
}


static int
qapplication_set_override_cursor(lua_State *L)
{
  QApplication::setOverrideCursor(luaQ_checkqvariant<QCursor>(L, 1));
  return 0;
}


static int
qapplication_change_override_cursor(lua_State *L)
{
  QApplication::changeOverrideCursor(luaQ_checkqvariant<QCursor>(L, 1));
  return 0;
}


static int
qapplication_restore_override_cursor(lua_State *L)
{
  QApplication::restoreOverrideCursor();
  return 0;
}

static int
qapplication_override_cursor(lua_State *L)
{
  QCursor *c = QApplication::overrideCursor();
  if (c)
    luaQ_pushqt(L, *c);
  else
    lua_pushnil(L);
  return 1;
}

static struct luaL_Reg qapplication_lib[] = {
  {"keyboardModifiers", qapplication_keyboard_modifiers},
  {"mouseButtons", qapplication_mouse_buttons},
  {"setOverrideCursor", qapplication_set_override_cursor},
  {"changeOverrideCursor", qapplication_change_override_cursor},
  {"restoreOverrideCursor", qapplication_restore_override_cursor},
  {"overrideCursor", qapplication_override_cursor},
  {0,0}
};

do_qhook(qapplication)




// ========================================
// QBRUSH


static int
qbrush_totable(lua_State *L)
{
  QBrush s = luaQ_checkqvariant<QBrush>(L, 1);
  lua_createtable(L, 0, 0);
  QMetaEnum m_style = f_enumerator("BrushStyle");
  Qt::BrushStyle style = s.style();
  f_pushflag(L, (int)style, m_style);
  lua_setfield(L, -2, "style");
  QVariant v;
  switch (style) 
    {
    case Qt::LinearGradientPattern:
    case Qt::ConicalGradientPattern:
    case Qt::RadialGradientPattern:
      v = qVariantFromValue<QGradient>(*s.gradient());
      luaQ_pushqt(L, v);
      lua_setfield(L, -2, "gradient");
      break;
    case Qt::TexturePattern:
      v = QVariant(s.textureImage());
      luaQ_pushqt(L, v);
      lua_setfield(L, -2, "texture");
      if (qVariantValue<QImage>(v).depth() > 1) 
        break;
    default:
      v = QVariant(s.color());
      luaQ_pushqt(L, v);
      lua_setfield(L, -2, "color");
    case Qt::NoBrush:
      break;
    }
  QTransform t = s.transform();
  if (t.isIdentity())
    return 1;
  luaQ_pushqt(L, QVariant(t));
  lua_setfield(L, -2, "transform");
  return 1;
}


static int
create_textured_brush(lua_State *L)
{
  QBrush brush(luaQ_checkqvariant<QImage>(L, 1));
  luaQ_pushqt(L, brush);
  return 1;
}


static int
qbrush_fromtable(lua_State *L)
{
  QBrush s;
  if (! lua_isnoneornil(L, 1)) 
    {
      luaL_checktype(L, 1, LUA_TTABLE);
      QMetaEnum m_style = f_enumerator("BrushStyle");
      const int t_gradient = qMetaTypeId<QGradient>();
      const int t_image = QMetaType::QImage;
      const int t_color = QMetaType::QColor;
      const int t_transform = QMetaType::QTransform;
      if (f_optvar(L, 1, "gradient", t_gradient))
        s = QBrush(qVariantValue<QGradient>
                   (luaQ_toqvariant(L, -1, t_gradient)));
      lua_pop(L, 1);
      if (f_optvar(L, 1, "texture", t_image))
        {
          // qt4.4 barks when creating an image texture 
          // brush outside the gui thread.
          // s=QBrush(qVariantValue<QImage>(luaQ_toqvariant(L,-1,t_image)));
          lua_pushcfunction(L, create_textured_brush); 
          lua_insert(L, -2);
          luaQ_call(L, 1, 1, 0);
          const int t_brush = QMetaType::QBrush;
          s = qVariantValue<QBrush>(luaQ_toqvariant(L, -1, t_brush));
        }
      lua_pop(L, 1);
      if (f_optflag(L, 1, "style", m_style))
        s.setStyle(Qt::BrushStyle(lua_tointeger(L, -1)));
      lua_pop(L, 1);
      if (f_optvar(L, 1, "color", t_color)) {
        if (s.style() == Qt::NoBrush) 
          s.setStyle(Qt::SolidPattern);
        s.setColor(qVariantValue<QColor>(luaQ_toqvariant(L, -1, t_color)));
      }
      lua_pop(L, 1);
      if (f_optvar(L, 1, "transform", t_transform))
        s.setTransform(qVariantValue<QTransform>
                       (luaQ_toqvariant(L, -1, t_transform)));
    lua_pop(L, 1);
    }
  luaQ_pushqt(L, QVariant(s));
  return 1;
}


do_luareg(qbrush)
do_hook(qbrush)



// ========================================
// QCOLOR


#define do_qcolor(do) \
  do ## flt("r",x=s.redF(),s[0]=x) \
  do ## flt("g",x=s.greenF(),s[1]=x) \
  do ## flt("b",x=s.blueF(),s[2]=x) \
  do ## optflt("a",x=s.alphaF(),s[3]=x)

do_totable(QColor,qcolor,do_qcolor)

static int
qcolor_fromtable(lua_State *L)
{
  QColor c;
  if (lua_gettop(L)>=3) {
    c.setRgbF(luaL_checknumber(L, 1),luaL_checknumber(L, 2),
              luaL_checknumber(L, 3),luaL_optnumber(L, 4, 1.0));
  } else if (lua_isstring(L,1)) {
    c.setNamedColor(lua_tostring(L,1));
  } else {
    qreal s[4] = {0,0,0,1};
    do_qcolor(fromtable_)
    c.setRgbF(s[0],s[1],s[2],s[3]);
  }
  luaQ_pushqt(L, QVariant(c));
  return 1;
}
  
do_luareg(qcolor)
do_hook(qcolor)



// ========================================
// QCOLORDIALOG


static int 
qcolordialog_getcolor(lua_State *L)
{
  int i = 1;
  QColor color;
  QVariant vcolor = luaQ_toqvariant(L, i);
  if (vcolor.userType() == qMetaTypeId<QColor>()) 
    color = luaQ_checkqvariant<QColor>(L, i++);
  QWidget *parent= luaQ_optqobject<QWidget>(L, i+1);
  color = QColorDialog::getColor(color, parent);
  if (color.isValid())
    luaQ_pushqt(L, color);
  else
    lua_pushnil(L);
  return 1;
}


static struct luaL_Reg qcolordialog_lib[] = {
  {"getColor", qcolordialog_getcolor },
  {0,0}
};

do_qhook(qcolordialog)





// ========================================
// QCURSOR


static QPixmap 
lua_checkpixmap(lua_State *L, int index)
{
  QVariant v = luaQ_toqvariant(L, index);
  if (v.userType() == qMetaTypeId<QImage>())
    return QPixmap::fromImage(qVariantValue<QImage>(v));
  if (v.userType() == qMetaTypeId<QPixmap>())
    return qVariantValue<QPixmap>(v);
  luaL_error(L, "illegal argument");
  return QPixmap();
}

static int
qcursor_new(lua_State *L)
{
  if (lua_isstring(L, 1))
    {
      QMetaEnum me = f_enumerator("CursorShape");
      int shape = me.keyToValue(lua_tostring(L, 1));
      if (shape < 0)
        luaL_error(L, "unrecognized cursor shape \"%s\".", lua_tostring(L, 1));
      luaQ_pushqt(L, QCursor(Qt::CursorShape(shape)));
      return 1;
    }
  else
    {
      QPixmap pm;
      QPixmap mask;
      int i = 1;
      pm = lua_checkpixmap(L, i++);
      if (lua_isuserdata(L, i))
        mask = lua_checkpixmap(L, i++);
      int hx = luaL_optinteger(L, i++, -1);
      int hy = luaL_optinteger(L, i++, -1);
      if (i <= 4)
        luaQ_pushqt(L, QCursor(pm , hx, hy));
      else if (pm.depth() == 1 && mask.depth() == 1)
        luaQ_pushqt(L, QCursor(QBitmap(pm), QBitmap(mask), hx, hy));
      else
        luaL_error(L, "expecting bitmap and mask of depth 1");
      return 1;
    }
}


static int
qcursor_hotspot(lua_State *L)
{
  QCursor cursor = luaQ_checkqvariant<QCursor>(L, 1);
  luaQ_pushqt(L, cursor.hotSpot());
  return 1;
}

static int
qcursor_mask(lua_State *L)
{
  QCursor cursor = luaQ_checkqvariant<QCursor>(L, 1);
  luaQ_pushqt(L, cursor.mask());
  return 1;
}

static int
qcursor_pixmap(lua_State *L)
{
  QCursor cursor = luaQ_checkqvariant<QCursor>(L, 1);
  luaQ_pushqt(L, cursor.pixmap());
  return 1;
}

static int
qcursor_shape(lua_State *L)
{
  QCursor cursor = luaQ_checkqvariant<QCursor>(L, 1);
  QMetaEnum me = f_enumerator("CursorShape");
  lua_pushstring(L, me.valueToKey(cursor.shape()));
  return 1;
}

static int
qcursor_pos(lua_State *L)
{
  luaQ_pushqt(L, QCursor::pos());
  return 1;
}

static int
qcursor_setpos(lua_State *L)
{
  QCursor::setPos(luaQ_checkqvariant<QPoint>(L, 1));
  return 0;
}

static struct luaL_Reg qcursor_lib[] = {
  {"new", qcursor_new},
  {"hotSpot", qcursor_hotspot},
  {"mask", qcursor_mask},
  {"pixmap", qcursor_pixmap},
  {"shape", qcursor_shape},
  {"pos", qcursor_pos},
  {"setPos", qcursor_setpos},
  {0,0}
};

do_qhook(qcursor)



// ========================================
// QDIALOG


static int 
qdialog_new(lua_State *L)
{
  QWidget *parent = luaQ_optqobject<QWidget>(L, 1);
  luaQ_pushqt(L, new QDialog(parent), !parent);
  return 1;
}

static int
qdialog_result(lua_State *L)
{
  QDialog *dlg = luaQ_checkqobject<QDialog>(L, 1);
  lua_pushinteger(L, dlg->result());
  return 1;
}

static int
qdialog_setresult(lua_State *L)
{
  QDialog *dlg = luaQ_checkqobject<QDialog>(L, 1);
  dlg->setResult(luaL_checkinteger(L, 2));
  return 0;
}

static struct luaL_Reg qdialog_lib[] = {
  {"new", qdialog_new},
  {"result", qdialog_result},
  {"setResult", qdialog_setresult},
  {0,0}
};

do_qhook(qdialog)


// ========================================
// QFILEDIALOG


typedef QFileDialog::Option QFDOption;
typedef QFileDialog::Options QFDOptions;

static QFDOption
f_checkfiledialogoption(lua_State *L, int index)
{
  if (!lua_isnoneornil(L, index))
    {
      QMetaEnum me = f_enumerator("Option", &QFileDialog::staticMetaObject);
      QByteArray b = luaQ_checkqvariant<QByteArray>(L, index);
      int v = me.keysToValue(b.constData());
      if (v > 0)
        return static_cast<QFDOption>(v);
    }
  luaL_error(L, "illegal file dialog options");
  return static_cast<QFDOption>(0);
}

static QFDOptions 
f_optfiledialogoptions(lua_State *L, int index, QFDOptions d)
{
  if (lua_isnoneornil(L, index))
    return d;
  QMetaEnum me = f_enumerator("Option", &QFileDialog::staticMetaObject);
  QByteArray b = luaQ_optqvariant<QByteArray>(L, index);
  int v = me.keysToValue(b.constData());
  if (v < 0)
    luaL_error(L, "illegal file dialog options");
  return static_cast<QFDOptions>(v);
}

static int 
qfiledialog_new(lua_State *L)
{
  QWidget *p = luaQ_optqobject<QWidget>(L, 1);
  QString c = luaQ_optqvariant<QString>(L, 2);
  QString d = luaQ_optqvariant<QString>(L, 3);
  QString f = luaQ_optqvariant<QString>(L, 4);  
  luaQ_pushqt(L, new QFileDialog(p,c,d,f), !p);
  return 1;
}

static int
qfiledialog_directory(lua_State *L)
{
  QFileDialog *dlg = luaQ_checkqobject<QFileDialog>(L, 1);
  luaQ_pushqt(L, dlg->directory().path());
  return 1;
}

#if QT_VERSION >= 0x40400
static int
qfiledialog_namefilters(lua_State *L)
{
  QFileDialog *dlg = luaQ_checkqobject<QFileDialog>(L, 1);
  luaQ_pushqt(L, dlg->nameFilters());
  return 0;
}
#endif

static int
qfiledialog_selectedfiles(lua_State *L)
{
  QVariantList v;
  QFileDialog *dlg = luaQ_checkqobject<QFileDialog>(L, 1);
  luaQ_pushqt(L, dlg->selectedFiles());
  return 1;
}

#if QT_VERSION >= 0x40400
static int
qfiledialog_selectednamefilter(lua_State *L)
{
  QVariantList v;
  QFileDialog *dlg = luaQ_checkqobject<QFileDialog>(L, 1);
  luaQ_pushqt(L, dlg->selectedNameFilter());
  return 1;
}
#endif

static int
qfiledialog_selectfile(lua_State *L)
{
  QFileDialog *dlg = luaQ_checkqobject<QFileDialog>(L, 1);
  dlg->selectFile(luaQ_checkqvariant<QString>(L, 2));
  return 0;
}

#if QT_VERSION >= 0x40400
static int
qfiledialog_selectnamefilter(lua_State *L)
{
  QFileDialog *dlg = luaQ_checkqobject<QFileDialog>(L, 1);
  dlg->selectNameFilter(luaQ_checkqvariant<QString>(L, 2));
  return 0;
}
#endif

static int
qfiledialog_setdirectory(lua_State *L)
{
  QFileDialog *dlg = luaQ_checkqobject<QFileDialog>(L, 1);
  dlg->setDirectory(luaQ_checkqvariant<QString>(L, 2));
  return 0;
}

#if QT_VERSION >= 0x40400
static int
qfiledialog_setnamefilters(lua_State *L)
{
  QFileDialog *dlg = luaQ_checkqobject<QFileDialog>(L, 1);
  QVariant v = luaQ_toqvariant(L, 2);
  if (v.userType() != QMetaType::QStringList)
    v = luaQ_checkqvariant<QString>(L, 2).split(";;");
  dlg->setNameFilters(v.toStringList());
  return 0;
}
#endif

#if QT_VERSION >= 0x40500
static int
qfiledialog_setoption(lua_State *L)
{
  QFileDialog *dlg = luaQ_checkqobject<QFileDialog>(L, 1);
  QFDOption opt = f_checkfiledialogoption(L, 2);
  bool b = lua_isnone(L, 3) || lua_toboolean(L, 3);
  dlg->setOption(opt, b);
  if (opt == QFileDialog::ShowDirsOnly)
    dlg->setFilter(b ? QDir::AllDirs : QDir::AllEntries);
  return 0;
}
#endif

#if QT_VERSION >= 0x40500
static int
qfiledialog_testoption(lua_State *L)
{
  QFileDialog *dlg = luaQ_checkqobject<QFileDialog>(L, 1);
  QFDOption opt = f_checkfiledialogoption(L, 2);
  lua_pushboolean(L, dlg->testOption(opt));
  return 1;
}
#endif

static int 
qfiledialog_getexistingdirectory(lua_State *L)
{
  QWidget *p = luaQ_optqobject<QWidget>(L, 1);
  QString c = luaQ_optqvariant<QString>(L, 2);
  QString d = luaQ_optqvariant<QString>(L, 3);
  QFDOptions o = f_optfiledialogoptions(L, 4, QFileDialog::ShowDirsOnly);
  o |= QFileDialog::DontUseNativeDialog;
  luaQ_pushqt(L, QFileDialog::getExistingDirectory(p,c,d,o));
  return 1;
}

static int 
qfiledialog_getopenfilename(lua_State *L)
{
  QWidget *p = luaQ_optqobject<QWidget>(L, 1);
  QString c = luaQ_optqvariant<QString>(L, 2);
  QString d = luaQ_optqvariant<QString>(L, 3);
  QString f = luaQ_optqvariant<QString>(L, 4);
  QString s = luaQ_optqvariant<QString>(L, 5);
  QFDOptions o = f_optfiledialogoptions(L, 6, 0);
  o |= QFileDialog::DontUseNativeDialog;
  luaQ_pushqt(L, QFileDialog::getOpenFileName(p,c,d,f,&s,o));
  luaQ_pushqt(L, s);
  return 2;
}

static int 
qfiledialog_getopenfilenames(lua_State *L)
{
  QWidget *p = luaQ_optqobject<QWidget>(L, 1);
  QString c = luaQ_optqvariant<QString>(L, 2);
  QString d = luaQ_optqvariant<QString>(L, 3);
  QString f = luaQ_optqvariant<QString>(L, 4);
  QString s = luaQ_optqvariant<QString>(L, 5);
  QFDOptions o = f_optfiledialogoptions(L, 6, 0);
  o |= QFileDialog::DontUseNativeDialog;
  luaQ_pushqt(L, QFileDialog::getOpenFileNames(p,c,d,f,&s,o));
  luaQ_pushqt(L, s);
  return 2;
}

static int 
qfiledialog_getsavefilename(lua_State *L)
{
  QWidget *p = luaQ_optqobject<QWidget>(L, 1);
  QString c = luaQ_optqvariant<QString>(L, 2);
  QString d = luaQ_optqvariant<QString>(L, 3);
  QString f = luaQ_optqvariant<QString>(L, 4);
  QString s = luaQ_optqvariant<QString>(L, 5);
  QFDOptions o = f_optfiledialogoptions(L, 6, 0);
  luaQ_pushqt(L, QFileDialog::getSaveFileName(p,c,d,f,&s,o));
  luaQ_pushqt(L, s);
  return 2;
}

static struct luaL_Reg qfiledialog_lib[] = {
  {"new", qfiledialog_new},
  {"directory", qfiledialog_directory},
  {"selectedFiles", qfiledialog_selectedfiles},
  {"selectFile", qfiledialog_selectfile},
  {"setDirectory", qfiledialog_setdirectory},
#if QT_VERSION >= 0x40400
  {"nameFilters", qfiledialog_namefilters},
  {"setNameFilters", qfiledialog_setnamefilters},
  {"selectNameFilter", qfiledialog_selectnamefilter},
  {"selectedNameFilter", qfiledialog_selectednamefilter},
#endif
#if QT_VERSION >= 0x40500
  {"setOption", qfiledialog_setoption},
  {"testOption", qfiledialog_testoption},  
#endif
  {"getExistingDirectory", qfiledialog_getexistingdirectory},
  {"getOpenFileName", qfiledialog_getopenfilename},
  {"getOpenFileNames", qfiledialog_getopenfilenames},
  {"getSaveFileName", qfiledialog_getsavefilename},
  {0,0}
};

do_qhook(qfiledialog)


// ========================================
// QFONT
  

#define do_qfont(do) \
  do ## optstr("family",x=s.family(),s.setFamily(x)) \
  do ## optint("size",x=s.pixelSize(), if (x>0) s.setPixelSize(x)) \
  do ## optint("pixelSize",x=s.pixelSize(), if (x>0) s.setPixelSize(x)) \
  do ## optflt("pointSize",x=s.pointSizeF(), if (x>0) s.setPointSizeF(x)) \
  do ## optbool("bold",x=s.bold(),if (x) s.setBold(true)) \
  do ## optbool("italic",x=s.italic(),if (x) s.setItalic(true)) \
  do ## optbool("underline",x=s.underline(),s.setUnderline(x)) \
  do ## optbool("overline",x=s.overline(), s.setOverline(x)) \
  do ## optbool("strikeOut",x=s.strikeOut(),s.setStrikeOut(x)) \
  do ## optbool("fixedPitch",x=s.fixedPitch(),s.setFixedPitch(x)) \
  do ## optbool("rawMode",x=s.rawMode(),s.setRawMode(x)) \
  do ## optint("weight",x=s.weight(),s.setWeight(x)) \
  do ## optint("stretch",x=s.stretch(),s.setStretch(x)) \
  do ## optbool("typewriter",x=(s.styleHint()==QFont::TypeWriter),\
                if (x) s.setStyleHint(QFont::TypeWriter,QFont::PreferMatch)) \
  do ## optbool("serif",x=(s.styleHint()==QFont::Serif),\
                if (x) s.setStyleHint(QFont::Serif))\
  do ## optbool("sans",x=(s.styleHint()==QFont::SansSerif),\
                if (x) s.setStyleHint(QFont::SansSerif))

static int
qfont_tostring(lua_State *L)
{
  QFont f = luaQ_checkqvariant<QFont>(L, 1);
  QString s = f.toString();
  lua_pushstring(L, s.toLocal8Bit().constData());
  return 1;
}

do_totable(QFont,qfont,do_qfont)

static int
qfont_fromtable(lua_State *L)
{
  QFont s;
  if (! lua_isstring(L, 1)) {
    s.setFamily("");
    do_qfont(fromtable_)
  } else
    s.fromString(QString::fromLocal8Bit(lua_tostring(L, 1)));
  luaQ_pushqt(L, QVariant(s));
  return 1;
}

static int
qfont_info(lua_State *L)
{
  QFont f = luaQ_checkqvariant<QFont>(L, 1);
  QFontInfo s = f;
  lua_createtable(L, 0, 2);
  totable_bool("bold", x=s.bold(),);
  totable_str("family", x=s.family(),);
  totable_bool("fixedPitch", x=s.fixedPitch(),);
  totable_bool("italic", x=s.italic(),);
  totable_flt("pointSize", x=s.pointSizeF(),);
  totable_int("pixelSize", x=s.pixelSize(),);
  totable_int("size", x=s.pixelSize(),);
  totable_bool("rawMode", x=s.rawMode(),);
  totable_int("weight", x=s.weight(),);
  totable_bool("typewriter", x=(s.styleHint()==QFont::TypeWriter),);
  totable_bool("serif", x=(s.styleHint()==QFont::Serif),);
  totable_bool("sans", x=(s.styleHint()==QFont::SansSerif),);
  // copied from font
  totable_bool("underline",x=f.underline(),);
  totable_bool("overline",x=f.overline(),);
  totable_bool("strikeOut",x=f.strikeOut(),);
  totable_int("stretch",x=f.stretch(),);
  // specific
  totable_bool("exactMatch", x=s.exactMatch(),);
  return 1;
}


static struct luaL_Reg qfont_lib[] = {
  {"tostring", qfont_tostring },
  {"totable", qfont_totable },
  {"new", qfont_fromtable }, 
  {"info", qfont_info },
  {0,0} 
}; 

do_hook(qfont)



// ========================================
// QFONTDIALOG


static int 
qfontdialog_getfont(lua_State *L)
{
  int i = 1;
  QFont font;
  QVariant vfont = luaQ_toqvariant(L, i);
  if (vfont.userType() == qMetaTypeId<QFont>()) 
    font = luaQ_checkqvariant<QFont>(L, i++);
  QWidget *parent= luaQ_optqobject<QWidget>(L, i+1);
  QString caption = luaQ_optqvariant<QString>(L, i+2);
  bool ok = false;
  if (caption.isNull())
    font = QFontDialog::getFont(&ok, font, parent);
  else
    font = QFontDialog::getFont(&ok, font, parent, caption);    
  if (ok)
    luaQ_pushqt(L, font);
  else
    lua_pushnil(L);
  return 1;
}


static struct luaL_Reg qfontdialog_lib[] = {
  {"getFont", qfontdialog_getfont },
  {0,0}
};

do_qhook(qfontdialog)




// ========================================
// QICON

static int
qicon_new(lua_State *L)
{
  QIcon icon;
  QVariant v = luaQ_toqvariant(L, 1);
  QVariant s = luaQ_toqvariant(L, 1, QMetaType::QString);
  if (v.userType() == QMetaType::QPixmap)
    icon = QIcon(qVariantValue<QPixmap>(v));
  else if (v.userType() == QMetaType::QImage)
    icon = QIcon(QPixmap::fromImage(qVariantValue<QImage>(v)));
  else if (s.userType() == QMetaType::QString)
    icon = QIcon(s.toString());
  else if (! lua_isnoneornil(L, 1))
    luaL_error(L, "bad argument #1 (string or image expected)");
  luaQ_pushqt(L, icon);
  return 1;
}

static struct luaL_Reg qicon_lib[] = {
  {"new", qicon_new },
  {0,0} 
}; 
  
do_qhook(qicon)




// ========================================
// QIMAGE


static int
qimage_new(lua_State *L)
{
  QImage image;
  if (lua_isuserdata(L, 1))
    {
      void *udata = luaL_checkudata(L, 2, LUA_FILEHANDLE);
      const char *format = luaL_optstring(L, 2, 0);
      QFile f;
      if (! f.open(*(FILE**)udata, QIODevice::ReadOnly))
        luaL_error(L,"cannot use stream for reading (%s)", 
                   f.errorString().toLocal8Bit().constData() );
      if (!image.load(&f,format) || image.isNull())
        luaL_error(L,"unable to load image file");
    }
  else if (lua_type(L, 1) == LUA_TSTRING)
    {
      const char *fname = lua_tostring(L, 1);
      const char *format = luaL_optstring(L, 2, 0);
      if (!image.load(QString::fromUtf8(fname),format) || image.isNull())
        luaL_error(L,"unable to load image file");
    }
  else
    {
      int w = luaL_checkinteger(L, 1);
      int h = luaL_checkinteger(L, 2);
      bool monochrome = lua_toboolean(L, 3);
      if (! lua_isnone(L, 3))
        luaL_checktype(L, 3, LUA_TBOOLEAN);
      if (monochrome)
        image = QImage(w, h, QImage::Format_Mono);
      else
        image = QImage(w, h, QImage::Format_ARGB32_Premultiplied);
    }
  luaQ_pushqt(L, QVariant(image));
  return 1;
}


static int
qimage_size(lua_State *L)
{
  QImage s = luaQ_checkqvariant<QImage>(L, 1);
  luaQ_pushqt(L, s.size());
  return 1;
}

static int
qimage_rect(lua_State *L)
{
  QImage s = luaQ_checkqvariant<QImage>(L, 1);
  luaQ_pushqt(L, s.rect());
  return 1;
}

static int
qimage_depth(lua_State *L)
{
  QImage s = luaQ_checkqvariant<QImage>(L, 1);
  lua_pushinteger(L, s.depth());
  return 1;
}

static int
qimage_topixmap(lua_State *L)
{
  QImage s = luaQ_checkqvariant<QImage>(L, 1);
  luaQ_pushqt(L, QPixmap::fromImage(s));
  return 1;
}

static int
qimage_save(lua_State *L)
{
  QImage s = luaQ_checkqvariant<QImage>(L, 1);
  QString fn = luaQ_optqvariant<QString>(L, 2);
  const char *format = 0;
  QFile f;
  if (fn.isEmpty() && lua_isuserdata(L, 2))
    {
      void *udata = luaL_checkudata(L, 2, LUA_FILEHANDLE);
      if (! f.open(*(FILE**)udata, QIODevice::WriteOnly))
        luaL_error(L,"cannot use stream for writing (%s)", 
                   f.errorString().toLocal8Bit().constData() );
      format = luaL_checkstring(L, 3);
    }
  else
    {
      f.setFileName(fn);
      QByteArray fname = fn.toLocal8Bit();
      if (! f.open(QIODevice::WriteOnly))
        luaL_error(L,"cannot open '%s'for writing (%s)", fname.constData(),
                   f.errorString().toLocal8Bit().constData() );
      format = strrchr(fname.constData(), '.');
      format = luaL_optstring(L, 3, (format) ? format+1 : 0);
    }
  QImageWriter writer(&f, format);
  if (! writer.write(s))
    {
      f.remove();
      if (writer.error() == QImageWriter::UnsupportedFormatError)
        luaL_error(L, "image format '%s' not supported for writing", format);
      else
        luaL_error(L, "error while writing file (%s)",
                   f.errorString().toLocal8Bit().constData() );
    }
  return 0;
}

static int
qimage_supportedformats(lua_State *L)
{
  const char *s = luaL_optstring(L, 1, "r");
  if ((s[0] != 'w' && s[0] != 'r') || (s[1] && s[1] != 'f'))
    luaL_error(L, "Illegal argument for qt.QImage.formats()");
  bool write = (s[0]=='w');
  bool filter = (s[1]=='f');
  QList<QByteArray> bl;
  if (write)
    bl = QImageWriter::supportedImageFormats();
  else
    bl = QImageReader::supportedImageFormats();
  QStringList sl;
  foreach(QByteArray b, bl)
    sl.append(QString::fromLocal8Bit(b));
  if (! filter)
    {
      luaQ_pushqt(L, sl);
      return 1;
    }
  else
    {
      QString f;
      foreach(QString s, sl)
        f.append(QString("%1 Files (*.%2);;")
                 .arg(s.toUpper()).arg(s.toLower()));
      f.append("All Files (*)");
      lua_pushstring(L, f.toLocal8Bit().constData());
      return 1;
    }
}



static luaL_Reg qimage_lib[] = {
  {"new", qimage_new},
  {"rect", qimage_rect},
  {"size", qimage_size},
  {"depth", qimage_depth},
  {"save", qimage_save},
  {0,0}
};


static luaL_Reg qimage_guilib[] = {
  {"topixmap", qimage_topixmap},
  {"formats", qimage_supportedformats},
  {0,0}
};


static int 
qimage_hook(lua_State *L)
{
  lua_getfield(L, -1, "__metatable");
  luaL_register(L, 0, qimage_lib);
  luaQ_register(L, qimage_guilib, QCoreApplication::instance());
  return 0;
}




// ========================================
// QKEYSEQUENCE


static int
qkeysequence_new(lua_State *L)
{
  QString s = luaQ_checkqvariant<QString>(L, 1);
  luaQ_pushqt(L, QKeySequence(s));
  return 1;
}

static int
qkeysequence_tostring(lua_State *L)
{
  QKeySequence k = luaQ_checkqvariant<QKeySequence>(L, 1);
  QByteArray s = k.toString().toLocal8Bit();
  lua_pushstring(L, s.constData());
  return 1;
}

static luaL_Reg qkeysequence_lib[] = {
  {"new", qkeysequence_new},
  {"tostring", qkeysequence_tostring},
  {0,0}
};

do_hook(qkeysequence)




// ========================================
// QMAINWINDOW

static int
qmainwindow_new(lua_State *L)
{
  QWidget *parent = luaQ_optqobject<QWidget>(L, 1);
  luaQ_pushqt(L, new QMainWindow(parent), !parent);
  return 1;
}

static int
qmainwindow_centralwidget(lua_State *L)
{
  QMainWindow *w = luaQ_checkqobject<QMainWindow>(L, 1);
  luaQ_pushqt(L, w->centralWidget());
  return 1;
}

static int
qmainwindow_menubar(lua_State *L)
{
  QMainWindow *w = luaQ_checkqobject<QMainWindow>(L, 1);
  luaQ_pushqt(L, w->menuBar());
  return 1;
}

static int
qmainwindow_setcentralwidget(lua_State *L)
{
  QMainWindow *w = luaQ_checkqobject<QMainWindow>(L, 1);
  QWidget *c = luaQ_checkqobject<QWidget>(L, 2);
  w->setCentralWidget(c);
  return 0;
}

static int
qmainwindow_setmenubar(lua_State *L)
{
  QMainWindow *w = luaQ_checkqobject<QMainWindow>(L, 1);
  QMenuBar *c = luaQ_checkqobject<QMenuBar>(L, 2);
  w->setMenuBar(c);
  return 0;
}

static int
qmainwindow_setstatusbar(lua_State *L)
{
  QMainWindow *w = luaQ_checkqobject<QMainWindow>(L, 1);
  QStatusBar *c = luaQ_checkqobject<QStatusBar>(L, 2);
  w->setStatusBar(c);
  return 0;
}

static int
qmainwindow_statusbar(lua_State *L)
{
  QMainWindow *w = luaQ_checkqobject<QMainWindow>(L, 1);
  luaQ_pushqt(L, w->statusBar());
  return 1;
}

static luaL_Reg qmainwindow_lib[] = {
  {"new", qmainwindow_new},
  {"centralWidget", qmainwindow_centralwidget},
  {"menuBar", qmainwindow_menubar},
  {"setCentralWidget", qmainwindow_setcentralwidget},
  {"setMenuBar", qmainwindow_setmenubar},
  {"setStatusBar", qmainwindow_setstatusbar},
  {"statusBar", qmainwindow_statusbar},
  {0,0}
};

do_qhook(qmainwindow)




// ========================================
// QMENU

static int
qmenu_new(lua_State *L)
{
  QWidget *parent = luaQ_optqobject<QWidget>(L, 1);
  QMenu *menu = new QMenu(parent);
  luaQ_pushqt(L, menu, !parent);
  return 1;
}

static int
qmenu_addLuaAction(lua_State *L)
{
  // qmenu:addLuaAction([icon,]text[,keysequence[,statustip]][function])
  int i = 1;
  QMenu *m = luaQ_checkqobject<QMenu>(L, i++);
  QIcon icon;
  QKeySequence keys;
  QString text;
  QString tip;
  if (luaQ_isqvariant<QIcon>(L, i))
    icon = luaQ_checkqvariant<QIcon>(L, i++);
  if (luaQ_isqvariant<QString>(L, i))
    text = luaQ_checkqvariant<QString>(L, i++);
  if (luaQ_isqvariant<QKeySequence>(L, i))
    keys = luaQ_checkqvariant<QKeySequence>(L, i++);
  else if (luaQ_isqvariant<QString>(L, i))
    keys = QKeySequence(luaQ_checkqvariant<QString>(L, i++));
  else if (lua_isnil(L, i))
    i++;
  if (luaQ_isqvariant<QString>(L, i))
    tip = luaQ_checkqvariant<QString>(L, i++);
  if (!lua_isnoneornil(L, i))
    luaL_checktype(L, i, LUA_TFUNCTION);
  if (text.isEmpty())
    luaL_error(L, "A string was expected");
  QAction *a = new QtLuaAction(luaQ_engine(L), m);
  a->setText(text);
  a->setIcon(icon);
  a->setShortcut(keys);
  a->setStatusTip(tip);
  m->addAction(a);
  if (lua_isfunction(L, i))
    luaQ_connect(L, a, SIGNAL(triggered(bool)), i);
  luaQ_pushqt(L, a);
  return 1;
}

static int
qmenu_addMenu(lua_State *L)
{
  QMenu *m = luaQ_checkqobject<QMenu>(L, 1);
  if (luaQ_isqobject<QMenu>(L, 2))
    luaQ_pushqt(L, m->addMenu(luaQ_checkqobject<QMenu>(L, 2)));
  else if (luaQ_isqvariant<QIcon>(L, 2))
    luaQ_pushqt(L, m->addMenu(luaQ_checkqvariant<QIcon>(L, 2),
                              luaQ_checkqvariant<QString>(L, 3) ));
  else
    luaQ_pushqt(L, m->addMenu(luaQ_checkqvariant<QString>(L, 2)));
  return 1;
}

static int
qmenu_addSeparator(lua_State *L)
{
  QMenu *m = luaQ_checkqobject<QMenu>(L, 1);
  luaQ_pushqt(L, m->addSeparator());
  return 1;
}

static int
qmenu_clear(lua_State *L)
{
  QMenu *m = luaQ_checkqobject<QMenu>(L, 1);
  m->clear();
  return 0;
}

static int
qmenu_exec(lua_State *L)
{
  QMenu *m = luaQ_checkqobject<QMenu>(L, 1);
  QPoint p = luaQ_optqvariant<QPoint>(L, 2, m->pos());
  QAction *a = luaQ_optqobject<QAction>(L, 3);
  luaQ_pushqt(L, m->exec(p, a));
  return 1;
}

static int
qmenu_insertMenu(lua_State *L)
{
  QMenu *m = luaQ_checkqobject<QMenu>(L, 1);
  QAction *before = luaQ_checkqobject<QAction>(L, 2);
  QMenu *menu = luaQ_checkqobject<QMenu>(L, 3);
  luaQ_pushqt(L, m->insertMenu(before, menu));
  return 1;
}

static int
qmenu_insertSeparator(lua_State *L)
{
  QMenu *m = luaQ_checkqobject<QMenu>(L, 1);
  QAction *before = luaQ_checkqobject<QAction>(L, 2);
  luaQ_pushqt(L, m->insertSeparator(before));
  return 1;
}

static int
qmenu_menuAction(lua_State *L)
{
  QMenu *m = luaQ_checkqobject<QMenu>(L, 1);
  luaQ_pushqt(L, m->menuAction());
  return 1;
}


static luaL_Reg qmenu_lib[] = {
  {"new", qmenu_new},
  {"addLuaAction", qmenu_addLuaAction},
  {"addMenu", qmenu_addMenu},
  {"addSeparator", qmenu_addSeparator},
  {"clear", qmenu_clear},
  {"exec", qmenu_exec},
  {"insertMenu", qmenu_insertMenu},
  {"insertSeparator", qmenu_insertSeparator},
  {"menuAction", qmenu_menuAction},
  {0,0}
};

do_qhook(qmenu)




// ========================================
// QMENUBAR


static int
qmenubar_new(lua_State *L)
{
  QWidget *parent = luaQ_optqobject<QWidget>(L, 1);
  QMenuBar *menu = new QMenuBar(parent);
  luaQ_pushqt(L, menu, !parent);
  return 1;
}

static int
qmenubar_addMenu(lua_State *L)
{
  QMenuBar *m = luaQ_checkqobject<QMenuBar>(L, 1);
  if (luaQ_isqobject<QMenu>(L, 2))
    luaQ_pushqt(L, m->addMenu(luaQ_checkqobject<QMenu>(L, 2)));
  else if (luaQ_isqvariant<QIcon>(L, 2))
    luaQ_pushqt(L, m->addMenu(luaQ_checkqvariant<QIcon>(L, 2),
                              luaQ_checkqvariant<QString>(L, 3) ));
  else
    luaQ_pushqt(L, m->addMenu(luaQ_checkqvariant<QString>(L, 2)));
  return 1;
}

static int
qmenubar_clear(lua_State *L)
{
  QMenuBar *m = luaQ_checkqobject<QMenuBar>(L, 1);
  m->clear();
  return 0;
}

static int
qmenubar_insertMenu(lua_State *L)
{
  QMenuBar *m = luaQ_checkqobject<QMenuBar>(L, 1);
  QAction *before = luaQ_checkqobject<QAction>(L, 2);
  QMenu *menu = luaQ_checkqobject<QMenu>(L, 3);
  luaQ_pushqt(L, m->insertMenu(before, menu));
  return 1;
}

static int
qmenubar_insertSeparator(lua_State *L)
{
  QMenuBar *m = luaQ_checkqobject<QMenuBar>(L, 1);
  QAction *before = luaQ_checkqobject<QAction>(L, 2);
  luaQ_pushqt(L, m->insertSeparator(before));
  return 1;
}

static int
qmenubar_addSeparator(lua_State *L)
{
  QMenuBar *m = luaQ_checkqobject<QMenuBar>(L, 1);
  luaQ_pushqt(L, m->addSeparator());
  return 1;
}

static luaL_Reg qmenubar_lib[] = {
  {"new", qmenubar_new},
  {"addMenu", qmenubar_addMenu},
  {"addSeparator", qmenubar_addSeparator},
  {"clear", qmenubar_clear},
  {"insertMenu", qmenubar_insertMenu},
  {"insertSeparator", qmenubar_insertSeparator},
  {0,0}
};

do_qhook(qmenubar)




// ========================================
// QPEN


static int
qpen_totable(lua_State *L)
{
  QPen s = luaQ_checkqvariant<QPen>(L, 1);
  lua_createtable(L, 0, 0);
  QMetaEnum m_penstyle = f_enumerator("PenStyle");
  QMetaEnum m_capstyle = f_enumerator("PenCapStyle");
  QMetaEnum m_joinstyle = f_enumerator("PenJoinStyle");
  Qt::PenStyle style = s.style();
  f_pushflag(L, (int)style, m_penstyle);
  lua_setfield(L, -2, "style");
  f_pushflag(L, (int)s.capStyle(), m_capstyle);
  lua_setfield(L, -2, "capStyle");
  f_pushflag(L, (int)s.joinStyle(), m_joinstyle);
  lua_setfield(L, -2, "joinStyle");
  luaQ_pushqt(L, QVariant(s.brush()));
  lua_setfield(L, -2, "brush"); 
  if (s.color().isValid())
    {
      luaQ_pushqt(L, QVariant(s.color()));
      lua_setfield(L, -2, "color");  
    }
  lua_pushnumber(L, s.widthF());
  lua_setfield(L, -2, "width");  
  lua_pushboolean(L, s.isCosmetic());
  lua_setfield(L, -2, "cosmetic");  
  if (style != Qt::NoPen && style != Qt::SolidLine)
    {
      lua_pushnumber(L, s.dashOffset());
      lua_setfield(L, -2, "dashOffset");  
    }
  if (s.joinStyle() == Qt::MiterJoin)
    {
      lua_pushnumber(L, s.miterLimit());
      lua_setfield(L, -2, "miterLimit");  
    }
  if (style == Qt::CustomDashLine)
    {
      QVector<qreal> v = s.dashPattern();
      lua_createtable(L, v.size(), 0);
      for (int i=0; i<v.size(); i++)
        {
          lua_pushnumber(L, v[i]);
          lua_rawseti(L, -2, i+1);
        }
      lua_setfield(L, -2, "dashPattern");
    }
  return 1;
}

static int
qpen_fromtable(lua_State *L)
{
  QPen s;
  if (! lua_isnoneornil(L, 1)) 
    {
      luaL_checktype(L, 1, LUA_TTABLE);
      QMetaEnum m_penstyle = f_enumerator("PenStyle");
      QMetaEnum m_capstyle = f_enumerator("PenCapStyle");
      QMetaEnum m_joinstyle = f_enumerator("PenJoinStyle");
      const int t_color = QMetaType::QColor;
      const int t_brush = QMetaType::QBrush;
      if (f_optflag(L, 1, "style", m_penstyle))
        s.setStyle(Qt::PenStyle(lua_tointeger(L, -1)));
      lua_pop(L, 1);
      if (f_optflag(L, 1, "capStyle", m_capstyle))
        s.setCapStyle(Qt::PenCapStyle(lua_tointeger(L, -1)));
      lua_pop(L, 1);
      if (f_optflag(L, 1, "joinStyle", m_joinstyle))
        s.setJoinStyle(Qt::PenJoinStyle(lua_tointeger(L, -1)));
      lua_pop(L, 1);
      if (f_optvar(L, 1, "color", t_color))
        s.setColor(qVariantValue<QColor>(luaQ_toqvariant(L, -1, t_color)));
      if (f_optvar(L, 1, "brush", t_brush))
        s.setBrush(qVariantValue<QBrush>(luaQ_toqvariant(L, -1, t_brush)));
      if (f_opttype(L, 1, "width", LUA_TNUMBER))
        s.setWidthF(lua_tonumber(L, -1));
      lua_pop(L, 1);
      if (f_opttype(L, 1, "cosmetic", LUA_TBOOLEAN))
        s.setCosmetic(lua_toboolean(L, -1));
      lua_pop(L, 1);
      if (f_opttype(L, 1, "miterLimit", LUA_TNUMBER))
        s.setMiterLimit(lua_tonumber(L, -1));
      lua_pop(L, 1);
      if (f_opttype(L, 1, "dashOffset", LUA_TNUMBER))
        s.setDashOffset(lua_tonumber(L, -1));
      lua_pop(L, 1);
      if (f_opttype(L, 1, "dashPattern", LUA_TTABLE))
        {
          QVector<qreal> v;
          int n = lua_objlen(L, -1);
          if (n & 1)
            luaL_error(L, "field 'dashPattern' must be an array with even length");
          for (int i=1; i<=n; i++)
            {
              lua_rawgeti(L, -1, i);
              v << lua_tonumber(L, -1);
              lua_pop(L, 1);
            }
          s.setDashPattern(v);
        }
      lua_pop(L, 1);
    }
  luaQ_pushqt(L, QVariant(s));
  return 1;
}

do_luareg(qpen)
do_hook(qpen)




// ========================================
// QTRANSFORM


#define do_qtransform(do) \
  do ## optflt("m11",x=s.m11(),m11=x) \
  do ## optflt("m12",x=s.m12(),m12=x) \
  do ## optflt("m13",x=s.m13(),m13=x) \
  do ## optflt("m21",x=s.m21(),m21=x) \
  do ## optflt("m22",x=s.m22(),m22=x) \
  do ## optflt("m23",x=s.m23(),m23=x) \
  do ## optflt("m31",x=s.m31(),m31=x) \
  do ## optflt("m32",x=s.m32(),m32=x) \
  do ## optflt("m33",x=s.m33(),m33=x) 


do_totable(QTransform,qtransform,do_qtransform)

static void
qtransform_getquad(lua_State *L, int k, QPolygonF &polygon)
{
  luaL_checktype(L, k, LUA_TTABLE);
  polygon.resize(4);
  for (int i=1; i<=4; i++) {
    lua_rawgeti(L, k, i);
    polygon[i-1] = luaQ_checkqvariant<QPointF>(L, -1);
    lua_pop(L, 1);
  }
}

static int
qtransform_fromtable(lua_State *L)
{
  QTransform c;
  if (lua_gettop(L) >= 2)
    {
      QPolygonF one, two;
      qtransform_getquad(L, 1, one);
      qtransform_getquad(L, 2, two);
      if (! QTransform::quadToQuad(one, two, c)) {
        lua_pushnil(L);
        return 1;
      }
    }
  else if (lua_gettop(L) >= 1)
    {
      qreal m11=0, m12=0, m13=0;
      qreal m21=0, m22=0, m23=0;
      qreal m31=0, m32=0, m33=0;
      do_qtransform(fromtable_);
      c.setMatrix(m11,m12,m13,m21,m22,m23,m31,m32,m33);
    }
  luaQ_pushqt(L, QVariant(c));
  return 1;
}


static int
qtransform_scaled(lua_State *L)
{
  QTransform c = luaQ_checkqvariant<QTransform>(L, 1);
  qreal x = luaL_checknumber(L, 2);
  qreal y = luaL_optnumber(L, 3, x);
  c.scale(x,y);
  luaQ_pushqt(L, QVariant(c));
  return 1;
}


static int
qtransform_translated(lua_State *L)
{
  QTransform c = luaQ_checkqvariant<QTransform>(L, 1);
  qreal x = luaL_checknumber(L, 2);
  qreal y = luaL_checknumber(L, 3);
  c.translate(x,y);
  luaQ_pushqt(L, QVariant(c));
  return 1;
}


static int
qtransform_sheared(lua_State *L)
{
  QTransform c = luaQ_checkqvariant<QTransform>(L, 1);
  qreal x = luaL_checknumber(L, 2);
  qreal y = luaL_checknumber(L, 3);
  c.shear(x,y);
  luaQ_pushqt(L, QVariant(c));
  return 1;
}


static int
qtransform_rotated(lua_State *L)
{
  static const char *theaxis[] = { "XAxis", "YAxis", "ZAxis", 0 };
  static const char *theunits[] = { "Degrees", "Radians", 0 };
  QTransform c = luaQ_checkqvariant<QTransform>(L, 1);
  qreal r = luaL_checknumber(L, 2);
  int axis = luaL_checkoption(L, 3, "ZAxis", theaxis);
  int unit = luaL_checkoption(L, 4, "Degrees", theunits);
  if (unit == 0)
    c.rotate(r, Qt::Axis(axis));
  else
    c.rotateRadians(r, Qt::Axis(axis));
  luaQ_pushqt(L, QVariant(c));
  return 1;
}

static int
qtransform_inverted(lua_State *L)
{
  bool invertible;
  QTransform c = luaQ_checkqvariant<QTransform>(L, 1);
  QTransform d = c.inverted(&invertible);
  if (invertible)
    luaQ_pushqt(L, QVariant(d));
  else
    lua_pushnil(L);
  return 1;
}

static int
qtransform_map(lua_State *L)
{
  QTransform c = luaQ_checkqvariant<QTransform>(L, 1);
  QVariant v = luaQ_toqvariant(L, 2);
  int type = v.userType();
  if (lua_isnumber(L, 2)) 
    {
        qreal tx, ty;
        qreal x = luaL_checknumber(L, 2);
        qreal y = luaL_checknumber(L, 3);
        c.map(x,y,&tx,&ty);
        lua_pushnumber(L, tx);
        lua_pushnumber(L, ty);
        return 2; 
    } 
#define DO(T,M) else if (type == qMetaTypeId<T>()) \
      luaQ_pushqt(L, qVariantFromValue<T>(c.M(qVariantValue<T>(v))))
  DO(QPoint,map);
  DO(QPointF,map);
  DO(QLine,map);
  DO(QLineF,map);
  DO(QPolygon,map);
  DO(QPolygonF,map);
  DO(QRegion,map);
  DO(QPainterPath,map);
  DO(QRect,mapRect);
  DO(QRectF,mapRect);
#undef DO
  else
    luaL_typerror(L, 2, "point, polygon, region, or path");
  return 1;
}


static luaL_Reg qtransform_lib[] = {
  {"totable", qtransform_totable},
  {"new", qtransform_fromtable},
  {"scaled", qtransform_scaled},
  {"translated", qtransform_translated},
  {"sheared", qtransform_sheared},
  {"rotated", qtransform_rotated},
  {"inverted", qtransform_inverted},
  {"map", qtransform_map},
  {0,0}
};


do_hook(qtransform)


// ========================================
// QWEBVIEW

#if HAVE_QTWEBKIT

static int 
qwebview_new(lua_State *L)
{
  QWidget *parent = luaQ_optqobject<QWidget>(L, 1);
  luaQ_pushqt(L, new QWebView(parent), !parent);
  return 1;
}

static int
qwebview_load(lua_State *L)
{
  QWebView *view = luaQ_checkqobject<QWebView>(L, 1);
  QUrl url = luaQ_checkqvariant<QUrl>(L, 2);
  view->load(url);
  return 0;
}

static int
qwebview_setcontent(lua_State *L)
{
  QWebView *view = luaQ_checkqobject<QWebView>(L, 1);
  QByteArray data = luaQ_checkqvariant<QByteArray>(L, 2);
  QString type = luaQ_checkqvariant<QString>(L, 3);
  QUrl url = luaQ_checkqvariant<QUrl>(L, 4);
  view->setContent(data, type, url);
  return 0;
}

static struct luaL_Reg qwebview_lib[] = {
  {"new", qwebview_new},
  {"load", qwebview_load},
  {"setContent", qwebview_setcontent},
  {0,0}
};

do_qhook(qwebview)

#endif

// ========================================
// QWIDGET


static int 
qwidget_new(lua_State *L)
{
  QWidget *parent = luaQ_optqobject<QWidget>(L, 1);
  QWidget *w = new QWidget(parent);
  luaQ_pushqt(L, w, !parent);
  return 1;
}

static int
qwidget_add_action(lua_State *L)
{
  QWidget *w = luaQ_checkqobject<QWidget>(L, 1);
  QAction *a = luaQ_checkqobject<QAction>(L, 2);
  w->addAction(a);
  return 0;
}


static int
qwidget_activate_window(lua_State *L)
{
  QWidget *w = luaQ_checkqobject<QWidget>(L, 1);
  w->activateWindow();
  return 0;
}

static int
qwidget_insert_action(lua_State *L)
{
  QWidget *w = luaQ_checkqobject<QWidget>(L, 1);
  QAction *b = luaQ_checkqobject<QAction>(L, 2);
  QAction *a = luaQ_checkqobject<QAction>(L, 3);
  w->insertAction(b, a);
  return 0;
}

static int
qwidget_remove_action(lua_State *L)
{
  QWidget *w = luaQ_checkqobject<QWidget>(L, 1);
  QAction *a = luaQ_checkqobject<QAction>(L, 2);
  w->removeAction(a);
  return 0;
}

static int
qwidget_actions(lua_State *L)
{
  QWidget *w = luaQ_checkqobject<QWidget>(L, 1);
  QVariantList vlist;
  QObjectPointer a;
  foreach(a, w->actions())
    vlist += qVariantFromValue(a);
  luaQ_pushqt(L, vlist);
  return 1;
}

static int
qwidget_map_to_global(lua_State *L)
{
  QWidget *w = luaQ_checkqobject<QWidget>(L, 1);
  QPoint p = luaQ_checkqvariant<QPoint>(L, 2);
  luaQ_pushqt(L, w->mapToGlobal(p));
  return 1;
}

static int
qwidget_map_from_global(lua_State *L)
{
  QWidget *w = luaQ_checkqobject<QWidget>(L, 1);
  QPoint p = luaQ_checkqvariant<QPoint>(L, 2);
  luaQ_pushqt(L, w->mapFromGlobal(p));
  return 1;
}

static int
qwidget_render(lua_State *L)
{
  QPainter *painter = 0;
  QPaintDevice *device = 0;
  QWidget *w = luaQ_checkqobject<QWidget>(L, 1);
  if (! lua_isnoneornil(L, 2))
    {
      QVariant v = luaQ_toqvariant(L, 2);
      if (v.userType() == qMetaTypeId<QPainter*>())
        painter = qVariantValue<QPainter*>(v);
      else if (v.userType() == qMetaTypeId<QPaintDevice*>())
        device = qVariantValue<QPaintDevice*>(v);
      else
        luaL_error(L, "Expecting QPainter* or QPaintDevice*");
    }
  if (painter)
    {
#if QT_VERSION >= 0x40400
      w->render(painter);
      lua_pushvalue(L, 2);
      return 0;
#else
      device = painter->device();
#endif
    }
  if (device)
    {
      w->render(device);
      lua_pushvalue(L, 2);
      return 0;
    }
  QImage image(w->size(), QImage::Format_ARGB32_Premultiplied);
  w->render(&image);
  luaQ_pushqt(L, image);
  return 1;
}

static void
name_2_attribute(lua_State *L, const char *name, 
                 Qt::WidgetAttribute &attrib)
{
  static struct { const char *n; int a; } s[] = {
    { "WA_Disabled", Qt::WA_Disabled },
    { "WA_UnderMouse", Qt::WA_UnderMouse },
    { "WA_MouseTracking", Qt::WA_MouseTracking },
    { "WA_ContentsPropagated", Qt::WA_ContentsPropagated },
    { "WA_OpaquePaintEvent", Qt::WA_OpaquePaintEvent },
    { "WA_NoBackground", Qt::WA_NoBackground },
    { "WA_StaticContents", Qt::WA_StaticContents },
    { "WA_LaidOut", Qt::WA_LaidOut },
    { "WA_PaintOnScreen", Qt::WA_PaintOnScreen },
    { "WA_NoSystemBackground", Qt::WA_NoSystemBackground },
    { "WA_UpdatesDisabled", Qt::WA_UpdatesDisabled },
    { "WA_Mapped", Qt::WA_Mapped },
    { "WA_MacNoClickThrough", Qt::WA_MacNoClickThrough },
    { "WA_PaintOutsidePaintEvent", Qt::WA_PaintOutsidePaintEvent },
    { "WA_InputMethodEnabled", Qt::WA_InputMethodEnabled },
    { "WA_WState_Visible", Qt::WA_WState_Visible },
    { "WA_WState_Hidden", Qt::WA_WState_Hidden },
    { "WA_ForceDisabled", Qt::WA_ForceDisabled },
    { "WA_KeyCompression", Qt::WA_KeyCompression },
    { "WA_PendingMoveEvent", Qt::WA_PendingMoveEvent },
    { "WA_PendingResizeEvent", Qt::WA_PendingResizeEvent },
    { "WA_SetPalette", Qt::WA_SetPalette },
    { "WA_SetFont", Qt::WA_SetFont },
    { "WA_SetCursor", Qt::WA_SetCursor },
    { "WA_NoChildEventsFromChildren", Qt::WA_NoChildEventsFromChildren },
    { "WA_WindowModified", Qt::WA_WindowModified },
    { "WA_Resized", Qt::WA_Resized },
    { "WA_Moved", Qt::WA_Moved },
    { "WA_PendingUpdate", Qt::WA_PendingUpdate },
    { "WA_InvalidSize", Qt::WA_InvalidSize },
    { "WA_MacBrushedMetal", Qt::WA_MacBrushedMetal },
    { "WA_MacMetalStyle", Qt::WA_MacMetalStyle },
    { "WA_CustomWhatsThis", Qt::WA_CustomWhatsThis },
    { "WA_LayoutOnEntireRect", Qt::WA_LayoutOnEntireRect },
    { "WA_OutsideWSRange", Qt::WA_OutsideWSRange },
    { "WA_GrabbedShortcut", Qt::WA_GrabbedShortcut },
    { "WA_TransparentForMouseEvents", Qt::WA_TransparentForMouseEvents },
    { "WA_PaintUnclipped", Qt::WA_PaintUnclipped },
    { "WA_SetWindowIcon", Qt::WA_SetWindowIcon },
    { "WA_NoMouseReplay", Qt::WA_NoMouseReplay },
    { "WA_DeleteOnClose", Qt::WA_DeleteOnClose },
    { "WA_RightToLeft", Qt::WA_RightToLeft },
    { "WA_SetLayoutDirection", Qt::WA_SetLayoutDirection },
    { "WA_NoChildEventsForParent", Qt::WA_NoChildEventsForParent },
    { "WA_ForceUpdatesDisabled", Qt::WA_ForceUpdatesDisabled },
    { "WA_WState_Created", Qt::WA_WState_Created },
    { "WA_WState_CompressKeys", Qt::WA_WState_CompressKeys },
    { "WA_WState_InPaintEvent", Qt::WA_WState_InPaintEvent },
    { "WA_WState_Reparented", Qt::WA_WState_Reparented },
    { "WA_WState_ConfigPending", Qt::WA_WState_ConfigPending },
    { "WA_WState_Polished", Qt::WA_WState_Polished },
    { "WA_WState_DND", Qt::WA_WState_DND },
    { "WA_WState_OwnSizePolicy", Qt::WA_WState_OwnSizePolicy },
    { "WA_WState_ExplicitShowHide", Qt::WA_WState_ExplicitShowHide },
    { "WA_ShowModal", Qt::WA_ShowModal },
    { "WA_MouseNoMask", Qt::WA_MouseNoMask },
    { "WA_GroupLeader", Qt::WA_GroupLeader },
    { "WA_NoMousePropagation", Qt::WA_NoMousePropagation },
    { "WA_Hover", Qt::WA_Hover },
    { "WA_InputMethodTransparent", Qt::WA_InputMethodTransparent },
    { "WA_QuitOnClose", Qt::WA_QuitOnClose },
    { "WA_KeyboardFocusChange", Qt::WA_KeyboardFocusChange },
    { "WA_AcceptDrops", Qt::WA_AcceptDrops },
    { "WA_DropSiteRegistered", Qt::WA_DropSiteRegistered },
    { "WA_ForceAcceptDrops", Qt::WA_ForceAcceptDrops },
    { "WA_WindowPropagation", Qt::WA_WindowPropagation },
    { "WA_NoX11EventCompression", Qt::WA_NoX11EventCompression },
    { "WA_TintedBackground", Qt::WA_TintedBackground },
    { "WA_X11OpenGLOverlay", Qt::WA_X11OpenGLOverlay },
    { "WA_AlwaysShowToolTips", Qt::WA_AlwaysShowToolTips },
    { "WA_MacOpaqueSizeGrip", Qt::WA_MacOpaqueSizeGrip },
    { "WA_SetStyle", Qt::WA_SetStyle },
    { "WA_SetLocale", Qt::WA_SetLocale },
    { "WA_MacShowFocusRect", Qt::WA_MacShowFocusRect },
    { "WA_MacNormalSize", Qt::WA_MacNormalSize },
    { "WA_MacSmallSize", Qt::WA_MacSmallSize },
    { "WA_MacMiniSize", Qt::WA_MacMiniSize },
    { "WA_LayoutUsesWidgetRect", Qt::WA_LayoutUsesWidgetRect },
    { "WA_StyledBackground", Qt::WA_StyledBackground },
    { "WA_MSWindowsUseDirect3D", Qt::WA_MSWindowsUseDirect3D },
    { "WA_CanHostQMdiSubWindowTitleBar", Qt::WA_CanHostQMdiSubWindowTitleBar },
    { "WA_MacAlwaysShowToolWindow", Qt::WA_MacAlwaysShowToolWindow },
    { "WA_StyleSheet", Qt::WA_StyleSheet },
#if QT_VERSION >= 0x40400
    { "WA_ShowWithoutActivating", Qt::WA_ShowWithoutActivating },
    { "WA_X11BypassTransientForHint", Qt::WA_X11BypassTransientForHint },
    { "WA_NativeWindow", Qt::WA_NativeWindow },
    { "WA_DontCreateNativeAncestors", Qt::WA_DontCreateNativeAncestors },
    { "WA_MacVariableSize", Qt::WA_MacVariableSize },
    { "WA_DontShowOnScreen", Qt::WA_DontShowOnScreen },
    { "WA_X11NetWmWindowTypeDesktop", Qt::WA_X11NetWmWindowTypeDesktop },
    { "WA_X11NetWmWindowTypeDock", Qt::WA_X11NetWmWindowTypeDock },
    { "WA_X11NetWmWindowTypeToolBar", Qt::WA_X11NetWmWindowTypeToolBar },
    { "WA_X11NetWmWindowTypeMenu", Qt::WA_X11NetWmWindowTypeMenu },
    { "WA_X11NetWmWindowTypeUtility", Qt::WA_X11NetWmWindowTypeUtility },
    { "WA_X11NetWmWindowTypeSplash", Qt::WA_X11NetWmWindowTypeSplash },
    { "WA_X11NetWmWindowTypeDialog", Qt::WA_X11NetWmWindowTypeDialog },
    { "WA_X11NetWmWindowTypeDropDownMenu", Qt::WA_X11NetWmWindowTypeDropDownMenu },
    { "WA_X11NetWmWindowTypePopupMenu", Qt::WA_X11NetWmWindowTypePopupMenu },
    { "WA_X11NetWmWindowTypeToolTip", Qt::WA_X11NetWmWindowTypeToolTip },
    { "WA_X11NetWmWindowTypeNotification", Qt::WA_X11NetWmWindowTypeNotification },
    { "WA_X11NetWmWindowTypeCombo", Qt::WA_X11NetWmWindowTypeCombo },
    { "WA_X11NetWmWindowTypeDND", Qt::WA_X11NetWmWindowTypeDND },
#endif
    { 0, 0 }
  };
  for(int i=0; s[i].n; i++)
    if (!strcmp(name, s[i].n))
      {
        attrib = Qt::WidgetAttribute(s[i].a);
        return;
      }
  luaL_error(L, "unrecognized widget attribute \"%s\"", name);
}


static void
name_2_window_flag(lua_State *L, const char *name, 
                   Qt::WindowFlags &flag, Qt::WindowFlags &mask)
{
  static struct { const char *n; int f,m; } s[] = {
    {"Widget", Qt::Widget, 0xff},
    {"Window", Qt::Window, 0xff},
    {"Dialog", Qt::Dialog, 0xff},
    {"Sheet", Qt::Sheet, 0xff},
    {"Drawer", Qt::Drawer, 0xff},
    {"Popup", Qt::Popup, 0xff},
    {"Tool", Qt::Tool, 0xff},
    {"ToolTip", Qt::ToolTip, 0xff},
    {"SplashScreen", Qt::SplashScreen, 0xff},
    {"Desktop", Qt::Desktop, 0xff},
    {"SubWindow", Qt::SubWindow, 0xff},
    {"MSWindowsFixedSizeDialogHint", Qt::MSWindowsFixedSizeDialogHint, 0},
    {"MSWindowsOwnDC", Qt::MSWindowsOwnDC, 0},
    {"X11BypassWindowManagerHint", Qt::X11BypassWindowManagerHint, 0},
    {"FramelessWindowHint", Qt::FramelessWindowHint, 0},
    {"WindowTitleHint", Qt::WindowTitleHint, 0},
    {"WindowSystemMenuHint", Qt::WindowSystemMenuHint, 0},
    {"WindowMinimizeButtonHint", Qt::WindowMinimizeButtonHint, 0},
    {"WindowMaximizeButtonHint", Qt::WindowMaximizeButtonHint, 0},
    {"WindowMinMaxButtonsHint", Qt::WindowMinMaxButtonsHint, 0},
    {"WindowContextHelpButtonHint", Qt::WindowContextHelpButtonHint, 0},
    {"WindowShadeButtonHint", Qt::WindowShadeButtonHint, 0},
    {"WindowStaysOnTopHint", Qt::WindowStaysOnTopHint, 0},
#if QT_VERSION >= 0x40400
    {"WindowOkButtonHint", Qt::WindowOkButtonHint, 0},
    {"WindowCancelButtonHint", Qt::WindowCancelButtonHint, 0},
#endif
    {"CustomizeWindowHint", Qt::CustomizeWindowHint, 0},
    {0,0,0}
  };
  for (int i=0; s[i].n; i++)
    if (! strcmp(name, s[i].n))
      {
        flag = Qt::WindowFlags(s[i].f);
        mask = Qt::WindowFlags(s[i].f | s[i].m);
        return;
      }
  luaL_error(L, "unrecognized window flag \"%s\"", name);
}

static int
qwidget_set_attribute(lua_State *L)
{

  QWidget *w = luaQ_checkqobject<QWidget>(L, 1);
  const char *name = luaL_checkstring(L, 2);
  bool value = lua_isnoneornil(L, 3) ? true : lua_toboolean(L, 3);
  Qt::WidgetAttribute attrib;
  name_2_attribute(L, name, attrib);
  w->setAttribute(attrib, value);
  return 0;
}

static int
qwidget_set_window_flag(lua_State *L)
{

  QWidget *w = luaQ_checkqobject<QWidget>(L, 1);
  const char *name = luaL_checkstring(L, 2);
  bool value = lua_isnoneornil(L, 3) ? true : lua_toboolean(L, 3);
  Qt::WindowFlags flag, mask;
  name_2_window_flag(L, name, flag, mask);
  Qt::WindowFlags f = w->windowFlags() & ~mask;
  if (value)
    w->setWindowFlags(f | flag);
  else
    w->setWindowFlags(f);
  return 0;
}

static int
qwidget_test_attribute(lua_State *L)
{

  QWidget *w = luaQ_checkqobject<QWidget>(L, 1);
  const char *name = luaL_checkstring(L, 2);
  Qt::WidgetAttribute attrib;
  name_2_attribute(L, name, attrib);
  lua_pushboolean(L, w->testAttribute(attrib));
  return 1;
}

static int
qwidget_test_window_flag(lua_State *L)
{

  QWidget *w = luaQ_checkqobject<QWidget>(L, 1);
  const char *name = luaL_checkstring(L, 2);
  Qt::WindowFlags flag, mask;
  name_2_window_flag(L, name, flag, mask);
  Qt::WindowFlags f = w->windowFlags();
  lua_pushboolean(L,  ((f & mask) == flag));
  return 1;
}

static int 
qwidget_window(lua_State *L)
{
  QWidget *w = luaQ_checkqobject<QWidget>(L, 1);
  luaQ_pushqt(L, w->window());
  return 1;
}

static struct luaL_Reg qwidget_lib[] = {
  {"new", qwidget_new},
  {"addAction", qwidget_add_action},
  {"activateWindow", qwidget_activate_window},
  {"insertAction", qwidget_insert_action},
  {"mapToGlobal", qwidget_map_to_global},
  {"mapFromGlobal", qwidget_map_from_global},
  {"removeAction", qwidget_remove_action},
  {"actions", qwidget_actions},
  {"render", qwidget_render},
  {"setAttribute", qwidget_set_attribute},
  {"setWindowFlag", qwidget_set_window_flag},
  {"window", qwidget_window},
  {"testAttribute", qwidget_test_attribute},
  {"testWindowFlag", qwidget_test_window_flag},
  {0,0}
};


do_qhook(qwidget)













// ====================================

LUA_EXTERNC QTGUI_API int
luaopen_libqtgui(lua_State *L)
{
  // load module 'qt'
  if (luaL_dostring(L, "require 'qt'"))
    lua_error(L);
  if (QApplication::type() == QApplication::Tty)
    luaL_error(L, "Graphics have been disabled (running with -nographics)");

  // register metatypes
  qRegisterMetaType<QGradient>("QGradient");
  qRegisterMetaType<QPainterPath>("QPainterPath");
  qRegisterMetaType<QPolygon>("QPolygon");
  qRegisterMetaType<QPolygonF>("QPolygonF");
  qRegisterMetaType<QPainter*>("QPainter*");
  qRegisterMetaType<QPrinter*>("QPrinter*");
  qRegisterMetaType<QPaintDevice*>("QPaintDevice*");

  // register object types
  QtLuaEngine::registerMetaObject(&QAction::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QtLuaAction::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QMainWindow::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QMenu::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QMenuBar::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QStatusBar::staticMetaObject);
  QtLuaEngine::registerMetaObject(&QWidget::staticMetaObject);

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
  
  HOOK_QOBJECT(QAction, qaction);  
  HOOK_QOBJECT(QApplication, qapplication);  
  HOOK_QVARIANT(QBrush, qbrush);
  HOOK_QVARIANT(QColor, qcolor);
  HOOK_QOBJECT(QColorDialog, qcolordialog);
  HOOK_QVARIANT(QCursor, qcursor);
  HOOK_QOBJECT(QDialog, qdialog);
  HOOK_QOBJECT(QFileDialog, qfiledialog);
  HOOK_QVARIANT(QFont, qfont);
  HOOK_QOBJECT(QFontDialog, qfontdialog);
  HOOK_QVARIANT(QIcon, qicon);
  HOOK_QVARIANT(QImage, qimage);
  HOOK_QVARIANT(QKeySequence, qkeysequence);
  HOOK_QOBJECT(QMainWindow, qmainwindow);
  HOOK_QOBJECT(QMenu, qmenu);
  HOOK_QOBJECT(QMenuBar, qmenubar);
  HOOK_QVARIANT(QPen, qpen);
  HOOK_QOBJECT(QtLuaAction, qtluaaction);  
  HOOK_QVARIANT(QTransform, qtransform);
#if HAVE_QTWEBKIT
  HOOK_QOBJECT(QWebView, qwebview);
#endif
  HOOK_QOBJECT(QWidget, qwidget);  

  return 0;
}



/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


