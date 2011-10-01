// -*- C++ -*-

#ifndef QTLUAENGINE_H
#define QTLUAENGINE_H

#include <QtGlobal>
#include <QByteArray>
#include <QList>
#include <QObject>
#include <QMetaObject>
#include <QMetaType>
#include <QPointer>
#include <QString>
#include <QStringList>
#include <QVariant>

#ifdef LUA_NOT_CXX
#include "lua.hpp"
#else
#include "lua.h"
#include "lauxlib.h"
#endif

#include "qtluaconf.h"

typedef QPointer<QObject> QObjectPointer;

Q_DECLARE_METATYPE(QObjectPointer)

class QTLUAAPI QtLuaEngine : public QObject
{
  Q_OBJECT
  Q_ENUMS(State);
  Q_PROPERTY(QByteArray lastErrorMessage READ lastErrorMessage)
  Q_PROPERTY(QStringList lastErrorLocation READ lastErrorLocation)
  Q_PROPERTY(bool printResults READ printResults WRITE setPrintResults)
  Q_PROPERTY(bool printErrors READ printErrors WRITE setPrintErrors)
  Q_PROPERTY(bool pauseOnError READ pauseOnError WRITE setPauseOnError)
  Q_PROPERTY(bool runSignalHandlers READ runSignalHandlers)
  Q_PROPERTY(State state READ state);
  Q_PROPERTY(bool ready READ isReady);
  Q_PROPERTY(bool running READ isRunning);
  Q_PROPERTY(bool paused READ isPaused);
public:
  QtLuaEngine(QObject *parent = 0);
  virtual ~QtLuaEngine();
  // meta objects
  static void registerMetaObject(const QMetaObject *mo);
  // named objects
  void nameObject(QObject *obj, QString name=QString());
  QObject *namedObject(QString name);
  QList<QObjectPointer> allNamedObjects();
  // state properties
  enum State { Ready, Running, Paused };
  State state() const;
  bool isReady() const { return state() == Ready; }
  bool isRunning() const { return state() == Running; }
  bool isPaused() const  { return state() == Paused; }
  bool isPausedOnError() const;
  // other properties
  QByteArray lastErrorMessage() const;
  QStringList lastErrorLocation() const;
  bool printResults() const;
  bool printErrors() const;
  bool pauseOnError() const;
  bool runSignalHandlers() const;
signals:
  void stateChanged(int state);
  void errorMessage(QByteArray message);
public slots:
  void setPrintResults(bool);
  void setPrintErrors(bool);
  void setPauseOnError(bool);
  bool stop(bool nopause=false);
  bool resume(bool nocontinue=false);
  bool eval(QByteArray s, bool async=false);
  bool eval(QString s, bool async=false);
  QVariantList evaluate(QByteArray s);
  QVariantList evaluate(QString s);
public:
  struct Global;
  struct Private;
  struct Unlocker;
  struct Catcher;
  struct Protector;
  struct Receiver;
  struct Receiver2;
private:
  Private *d;
  lua_State *L;
  friend class QtLuaLocker;
};


class QTLUAAPI QtLuaLocker
{
public:
  explicit QtLuaLocker(QtLuaEngine *engine);
  explicit QtLuaLocker(QtLuaEngine *engine, int timeOut);
  ~QtLuaLocker();
  void setRunning();
  bool isReady()        { return (count>0) && engine->isReady(); }
  bool isPaused()       { return (count>0) && engine->isPaused(); }
  operator lua_State*() { return (count>0) ? engine->L : 0; }
private:
  Q_DISABLE_COPY(QtLuaLocker)
  QtLuaEngine *engine;
  int count;
};


QTLUAAPI QtLuaEngine *luaQ_engine(lua_State *L);
QTLUAAPI QVariant luaQ_toqvariant(lua_State *L, int i, int type = 0);
QTLUAAPI QObject* luaQ_toqobject(lua_State *L, int i, const QMetaObject *m=0);
QTLUAAPI void luaQ_pushqt(lua_State *L);
QTLUAAPI void luaQ_pushqt(lua_State *L, const QVariant &var);
QTLUAAPI void luaQ_pushqt(lua_State *L, QObject *obj, bool owned=false);
QTLUAAPI void luaQ_pushmeta(lua_State *L, int type);
QTLUAAPI void luaQ_pushmeta(lua_State *L, const QMetaObject *mo);
QTLUAAPI void luaQ_pushmeta(lua_State *L, const QObject *o);
QTLUAAPI bool luaQ_connect(lua_State *L, QObject *o, const char *s, int fi,
                           bool direct=false);
QTLUAAPI bool luaQ_disconnect(lua_State *L, QObject*o, const char *s, int fi);
QTLUAAPI void luaQ_doevents(lua_State *L, bool wait);
QTLUAAPI void luaQ_pause(lua_State *L);
QTLUAAPI void luaQ_resume(lua_State *L, bool nocontinue=false);
QTLUAAPI void luaQ_call(lua_State *L, int na, int nr, QObject *obj=0);
QTLUAAPI int  luaQ_pcall(lua_State *L, int na, int nr, int eh, QObject *obj=0);
QTLUAAPI void luaQ_register(lua_State *L, const luaL_Reg *l, QObject *obj);

template<typename T> inline bool
luaQ_isqobject(lua_State *L, int index, T* = 0)
{
  T *obj = qobject_cast<T*>(luaQ_toqobject(L, index));
  return (obj != 0);
}

template<typename T> inline bool
luaQ_isqvariant(lua_State *L, int index, T* = 0)
{
  int type = qMetaTypeId<T>();
  QVariant v = luaQ_toqvariant(L, index, type);
  return (v.userType() == type);
}

template<typename T> inline T*
luaQ_checkqobject(lua_State *L, int index, T* = 0)
{
  T *obj = qobject_cast<T*>(luaQ_toqobject(L, index));
  if (! obj) 
    luaL_typerror(L, index, T::staticMetaObject.className());
  return obj;
}

template<typename T> inline T
luaQ_checkqvariant(lua_State *L, int index, T* = 0)
{
  int type = qMetaTypeId<T>();
  QVariant v = luaQ_toqvariant(L, index, type);
  if (v.userType() != type)
    luaL_typerror(L, index, QMetaType::typeName(type));
  return qVariantValue<T>(v);
}

template<typename T> inline T*
luaQ_optqobject(lua_State *L, int index, T *d = 0)
{
  if (!lua_isnoneornil(L, index))
    return luaQ_checkqobject<T>(L, index);
  return d;
}

template<typename T> inline T
luaQ_optqvariant(lua_State *L, int index, T d = T())
{
  if (!lua_isnoneornil(L, index))
    return luaQ_checkqvariant<T>(L, index);
  return d;
}

#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


