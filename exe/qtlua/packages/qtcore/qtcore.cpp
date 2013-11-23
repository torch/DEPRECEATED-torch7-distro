// -*- C++ -*-

#include "qtcore.h"

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>

#include <QCoreApplication>
#include <QLibrary>
#include <QLine>
#include <QLineF>
#include <QMetaMethod>
#include <QMetaObject>
#include <QMetaType>
#include <QObject>
#include <QPoint>
#include <QPointF>
#include <QRect>
#include <QRectF>
#include <QStringList>
#include <QSize>
#include <QSizeF>
#include <QTimer>
#include <QUrl>
#include <QVariant>


// ========================================
// UTILITIES 


static void
f_checktype(lua_State *L, int index, const char *name, int type)
{ 
  if (index)
    lua_getfield(L,index,name); 
  int t = lua_type(L, -1);
  if (t != type)
    luaL_error(L, "%s expected in field " LUA_QS ", got " LUA_QS,
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

#define do_all(T,t,do) \
  do_totable(T,t,do) \
  do_fromtable(T,t,do,T s,s) \
  do_luareg(t) \
  do_hook(t)



// ========================================
// QBYTEARRAY


static int
qbytearray_totable(lua_State *L)
{
  QByteArray s = luaQ_checkqvariant<QByteArray>(L, 1);
  lua_createtable(L, s.size(), 0);
  for (int i=0; i<s.size(); i++)
    {
      lua_pushinteger(L, s.at(i));
      lua_rawseti(L, -2, i+1);
    }
  return 1;
}


static int
qbytearray_new(lua_State *L)
{
  QByteArray s;
  if (lua_istable(L, 1))
    {
      int n = lua_objlen(L, 1);
      s.reserve(n);
      for (int i=1; i<n; i++)
        {
          lua_rawgeti(L, 1, i);
          s.append((char)(lua_tointeger(L, -1)));
          lua_pop(L, 1);
        }
      s.squeeze();
    }
  else
    s = luaQ_checkqvariant<QByteArray>(L, 1);
  luaQ_pushqt(L, s);
  return 1;
}


static const luaL_Reg qbytearray_lib[] = {
  {"totable", qbytearray_totable},
  {"new", qbytearray_new},
  {0, 0}
};


static int
qbytearray_hook(lua_State *L)
{
  lua_getfield(L, -1, "__metatable");
  luaL_register(L, 0, qbytearray_lib);
  return 0;
}



// ========================================
// QCOREAPPLICATION

static int
qcoreapplication_instance(lua_State *L)
{
  luaQ_pushqt(L, QCoreApplication::instance());
  return 1;
}

static const luaL_Reg qcoreapplication_lib[] = {
  {"new", qcoreapplication_instance},
  {0, 0}
};

do_qhook(qcoreapplication)



// ========================================
// QLINE, QLINEF

#define do_qline(do) \
  do ## int("x1",x=s.x1(),s[0]=x) \
  do ## int("y1",x=s.y1(),s[1]=x) \
  do ## int("x2",x=s.x2(),s[2]=x) \
  do ## int("y2",x=s.y2(),s[3]=x)

do_totable(QLine,qline,do_qline)
do_fromtable(QLine,qline,do_qline,
             int s[4];s[0]=s[1]=s[2]=s[3]=0,
             QLine(s[0],s[1],s[2],s[3]))
do_luareg(qline)
do_hook(qline)


#define do_qlinef(do) \
  do ## flt("x1",x=s.x1(),s[0]=x) \
  do ## flt("y1",x=s.y1(),s[1]=x) \
  do ## flt("x2",x=s.x2(),s[2]=x) \
  do ## flt("y2",x=s.y2(),s[3]=x)

do_totable(QLineF,qlinef,do_qlinef)
do_fromtable(QLineF,qlinef,do_qlinef,
             qreal s[4];s[0]=s[1]=s[2]=s[3]=0,
             QLineF(s[0],s[1],s[2],s[3]))
do_luareg(qlinef)
do_hook(qlinef)



// ========================================
// QOBJECT


static int qobject_parent(lua_State *L)
{
  QObject *w = luaQ_checkqobject<QObject>(L, 1);
  luaQ_pushqt(L, w->parent());
  return 1;
}


static int qobject_children(lua_State *L)
{
  QObject *w = luaQ_checkqobject<QObject>(L, 1);
  QVariantList v;
  QObjectPointer p;
  foreach(p, w->children())
    v << qVariantFromValue<QObjectPointer>(p);
  luaQ_pushqt(L, v);
  return 1;
}


static int qobject_dumpobjectinfo(lua_State *L)
{
  QObject *w = luaQ_checkqobject<QObject>(L, 1);
  w->dumpObjectInfo();
  return 0;
}


static int qobject_dumpobjecttree(lua_State *L)
{
  QObject *w = luaQ_checkqobject<QObject>(L, 1);
  w->dumpObjectTree();
  return 0;
}


static struct luaL_Reg qobject_lib[] = {
  {"parent", qobject_parent},
  {"children", qobject_children},
  {"dumpObjectInfo", qobject_dumpobjectinfo},
  {"dumpObjectTree", qobject_dumpobjecttree},
  {0,0}
};


do_qhook(qobject)



// ========================================
// QPOINT, QPOINTF

#define do_qpoint(do) \
  do ## int("x",x=s.x(),s.rx()=x) \
  do ## int("y",x=s.y(),s.ry()=x)

do_all(QPoint,qpoint,do_qpoint)


#define do_qpointf(do) \
  do ## flt("x",x=s.x(),s.rx()=x) \
  do ## flt("y",x=s.y(),s.ry()=x)

do_all(QPointF,qpointf,do_qpointf)


// ========================================
// QRECT, QRECTF

#define do_qrect(do) \
  do ## int("x",x=s.x(),s.setLeft(x)) \
  do ## int("y",x=s.y(),s.setTop(x)) \
  do ## int("width",x=s.width(),s.setWidth(x)) \
  do ## int("height",x=s.height(),s.setHeight(x))

do_all(QRect,qrect,do_qrect)


#define do_qrectf(do) \
  do ## flt("x",x=s.x(),s.setLeft(x)) \
  do ## flt("y",x=s.y(),s.setTop(x)) \
  do ## flt("width",x=s.width(),s.setWidth(x)) \
  do ## flt("height",x=s.height(),s.setHeight(x))

do_all(QRectF,qrectf,do_qrectf)




// ========================================
// QSIZE, QSIZEF


#define do_qsize(do) \
  do ## int("width",x=s.width(),s.rwidth()=x) \
  do ## int("height",x=s.height(),s.rheight()=x)

do_all(QSize,qsize,do_qsize)


#define do_qsizef(do) \
  do ## flt("width",x=s.width(),s.rwidth()=x) \
  do ## flt("height",x=s.height(),s.rheight()=x)

do_all(QSizeF,qsizef,do_qsizef)


// ========================================
// QSTRING


static int
qstring_totable(lua_State *L)
{
  QString s = luaQ_checkqvariant<QString>(L, 1);
  lua_createtable(L, s.size(), 0);
  for (int i=0; i<s.size(); i++)
    {
      lua_pushinteger(L, s.at(i).unicode());
      lua_rawseti(L, -2, i+1);
    }
  return 1;
}


static int
qstring_new(lua_State *L)
{
  QString s;
  if (lua_istable(L, 1))
    {
      int n = lua_objlen(L, 1);
      s.reserve(n);
      for (int i=1; i<n; i++)
        {
          lua_rawgeti(L, 1, i);
          s += QChar((ushort)(lua_tointeger(L, -1)));
          lua_pop(L, 1);
        }
      s.squeeze();
    }
  else
    s = luaQ_checkqvariant<QString>(L, 1);
  luaQ_pushqt(L, s);
  return 1;
}


static const luaL_Reg qstring_lib[] = {
  {"totable", qstring_totable},
  {"new", qstring_new},
  {0, 0}
};


static int
qstring_hook(lua_State *L)
{
  lua_getfield(L, -1, "__metatable");
  luaL_register(L, 0, qstring_lib);
  return 0;
}



// ========================================
// QSTRINGLIST


static int
qstringlist_totable(lua_State *L)
{
  QStringList l = luaQ_checkqvariant<QStringList>(L, 1);
  lua_createtable(L, l.size(), 0);
  for (int i=0; i<l.size(); i++)
    {
      luaQ_pushqt(L, QVariant(l[i]));
      lua_rawseti(L, -2, i+1);
    }
  return 1;
}


static int
qstringlist_new(lua_State *L)
{
  QStringList l;
  if (! lua_isnone(L, 1))
    {
      luaL_checktype(L, 1, LUA_TTABLE);
      int n = lua_objlen(L, 1);
      for (int i=1; i<=n; i++)
        {
          lua_rawgeti(L, 1, i);
          QVariant v = luaQ_toqvariant(L, -1, QMetaType::QString);
          if (v.userType() != QMetaType::QString)
            luaL_error(L, "table element cannot be converted to a QString");
          l += v.toString();
          lua_pop(L, 1);
        }
    }
  luaQ_pushqt(L, QVariant(l));
  return 1;
}


static const luaL_Reg qstringlist_lib[] = {
  {"totable", qstringlist_totable},
  {"new", qstringlist_new},
  {0, 0}
};


static int
qstringlist_hook(lua_State *L)
{
  lua_getfield(L, -1, "__metatable");
  luaL_register(L, 0, qstringlist_lib);
  return 0;
}


// ========================================
// QTIMER

static int
qtimer_new(lua_State *L)
{
  QObject *parent = luaQ_optqobject<QObject>(L, 1);
  QTimer *t = new QTimer(parent);
  luaQ_pushqt(L, t, !parent);
  return 1;
}

static const luaL_Reg qtimer_lib[] = {
  {"new", qtimer_new},
  {0,0}
};

do_hook(qtimer)


// ========================================
// QURL

static int
qurl_tostring(lua_State *L)
{
  QUrl url = luaQ_checkqvariant<QUrl>(L, 1);
  lua_pushstring(L, url.toEncoded().constData());
  return 1;
}


static int
qurl_new(lua_State *L)
{
  QString s = luaQ_checkqvariant<QString>(L, 1);
  luaQ_pushqt(L, QUrl(s));
  return 1;
}

static int
qurl_tolocalfile(lua_State *L)
{
  QUrl url = luaQ_checkqvariant<QUrl>(L, 1);
#if QT_VERSION >= 0x040700
  if (!url.isLocalFile())
    lua_pushnil(L);
  else
#endif
    lua_pushstring(L, url.toLocalFile().toLocal8Bit().constData());
  return 1;
}

static int
qurl_fromlocalfile(lua_State *L)
{
  QString s = luaQ_checkqvariant<QString>(L, 1);
  luaQ_pushqt(L, QUrl::fromLocalFile(s));
  return 1;
}


static const luaL_Reg qurl_lib[] = {
  {"tostring", qurl_tostring},
  {"new", qurl_new},
  {"tolocalfile", qurl_tolocalfile},
  {"fromlocalfile", qurl_fromlocalfile},
  {0, 0}
};


static int
qurl_hook(lua_State *L)
{
  lua_getfield(L, -1, "__metatable");
  luaL_register(L, 0, qurl_lib);
  return 0;
}


// ========================================
// QVARIANTLIST


static int
qvariantlist_totable(lua_State *L)
{
  QVariantList l = luaQ_checkqvariant<QVariantList>(L, 1);
  lua_createtable(L, l.size(), 0);
  for (int i=0; i<l.size(); i++)
    {
      luaQ_pushqt(L, QVariant(l[i]));
      lua_rawseti(L, -2, i+1);
    }
  return 1;
}


static int
qvariantlist_new(lua_State *L)
{
  QVariantList l;
  if (! lua_isnone(L, 1))
    {
      luaL_checktype(L, 1, LUA_TTABLE);
      int n = lua_objlen(L, 1);
      for (int i=1; i<=n; i++)
        {
          lua_rawgeti(L, 1, i);
          l += luaQ_toqvariant(L, -1);
          lua_pop(L, 1);
        }
    }
  luaQ_pushqt(L, QVariant(l));
  return 1;
}


static const luaL_Reg qvariantlist_lib[] = {
  {"totable", qvariantlist_totable},
  {"new", qvariantlist_new},
  {0, 0}
};


static int
qvariantlist_hook(lua_State *L)
{
  lua_getfield(L, -1, "__metatable");
  luaL_register(L, 0, qvariantlist_lib);
  return 0;
}



// ========================================
// QVARIANTMAP


static int
qvariantmap_totable(lua_State *L)
{
  QVariantMap m = luaQ_checkqvariant<QVariantMap>(L, 1);
  lua_createtable(L, 0, 0);
  for (QVariantMap::const_iterator it=m.constBegin(); it!=m.constEnd(); ++it)
    {
      lua_pushstring(L, it.key().toLocal8Bit().constData());
      luaQ_pushqt(L, it.value());
      lua_rawset(L, -3);
    }
  return 1;
}


static int
qvariantmap_new(lua_State *L)
{
  QVariantMap l;
  if (! lua_isnone(L, 1))
    {
      luaL_checktype(L, 1, LUA_TTABLE);
      lua_pushnil(L);
      while (lua_next(L, -2))
        {
          QVariant k = luaQ_toqvariant(L, -2, QMetaType::QString);
          if (k.userType() != QMetaType::QString)
            luaL_error(L, "table element cannot be converted to a QString");
          l[k.toString()] = luaQ_toqvariant(L, -1);
          lua_pop(L, 1);
        }
    }
  luaQ_pushqt(L, QVariant(l));
  return 1;
}


static const luaL_Reg qvariantmap_lib[] = {
  {"totable", qvariantmap_totable},
  {"new", qvariantmap_new},
  {0, 0}
};


static int
qvariantmap_hook(lua_State *L)
{
  lua_getfield(L, -1, "__metatable");
  luaL_register(L, 0, qvariantmap_lib);
  return 0;
}














// ====================================

LUA_EXTERNC QTCORE_API int
luaopen_libqtcore(lua_State *L)
{
  // load module 'qt'
  if (luaL_dostring(L, "require 'qt'"))
    lua_error(L);

  // hooks for objects
#define HOOK_QOBJECT(T, t) \
     lua_pushcfunction(L, t ## _hook);\
     luaQ_pushmeta(L, &T::staticMetaObject);\
     lua_call(L, 1, 0);
  
  // hooks for qvariants
#define HOOK_QVARIANT(T, t) \
     lua_pushcfunction(L, t ## _hook);\
     luaQ_pushmeta(L, QMetaType::T);\
     lua_call(L, 1, 0)

  
  HOOK_QVARIANT(QByteArray, qbytearray);
  HOOK_QOBJECT(QCoreApplication, qcoreapplication);
  HOOK_QVARIANT(QLine,qline);
  HOOK_QVARIANT(QLineF,qlinef);
  HOOK_QOBJECT (QObject, qobject)
  HOOK_QVARIANT(QPoint,qpoint);
  HOOK_QVARIANT(QPointF,qpointf);
  HOOK_QVARIANT(QRect,qrect);
  HOOK_QVARIANT(QRectF,qrectf);
  HOOK_QVARIANT(QSize,qsize);
  HOOK_QVARIANT(QSizeF,qsizef);
  HOOK_QVARIANT(QString, qstring);
  HOOK_QVARIANT(QStringList, qstringlist);
  HOOK_QOBJECT (QTimer, qtimer)
  HOOK_QVARIANT(QUrl,qurl);
  HOOK_QVARIANT(QVariantList, qvariantlist);
  HOOK_QVARIANT(QVariantMap, qvariantmap);

  return 0;
}



/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


