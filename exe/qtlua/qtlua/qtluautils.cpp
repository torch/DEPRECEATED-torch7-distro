// -*- C++ -*-

#include "qtluautils.h"
#include "qtluaengine.h"

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
// More utilities



static int
luaQ_gettable(struct lua_State *L)
{
  luaL_checktype(L, -2, LUA_TTABLE);
  lua_gettable(L, -2);
  return 1;
}


/*! Same as lua_getfield but returns \a nil
  if an error occurs while processing the 
  metatable \a __index event. */

void
luaQ_getfield(struct lua_State *L, int index, const char *name)
{
  lua_pushvalue(L, index);
  lua_pushcfunction(L, luaQ_gettable);
  lua_insert(L, -2);
  lua_pushstring(L, name);
  lua_Hook hf = lua_gethook(L);
  int hm = lua_gethookmask(L);
  int hc = lua_gethookcount(L);
  lua_sethook(L, 0, 0, 0);
  if (lua_pcall(L, 2, 1, 0))
    {
      lua_pop(L, 1);
      lua_pushnil(L);
    }
  lua_sethook(L, hf, hm, hc);
}





// ========================================
// Error handlers




/*! Enrichs the message on the stack with a traceback.
  This function must be called with one string argument
  representing an error message for instance.
  It safely augments the string with a stack trace
  starting \a skip levels above the current level.
  This function does not cause an error.
  If something goes wrong during the stack exploration,
  the function silently returns the unchanged error message.
 */

int
luaQ_tracebackskip(struct lua_State *L, int skip)
{
  // stack: msg
  luaQ_getfield(L, LUA_GLOBALSINDEX, "debug");
  luaQ_getfield(L, -1, "traceback");
  // stack: traceback debug msg
  lua_remove(L, -2);
  // stack: traceback msg
  if (! lua_isfunction(L, -1)) 
    {
      lua_pop(L, 1);
    } 
  else 
    {
      lua_pushvalue(L, -2);
      lua_pushinteger(L, skip+1);
      // save hook
      lua_Hook hf = lua_gethook(L);
      int hm = lua_gethookmask(L);
      int hc = lua_gethookcount(L);
      lua_sethook(L, 0, 0, 0);
      // stack: skip msg traceback msg
      if (lua_pcall(L, 2, 1, 0))
        // stack: err msg
        lua_remove(L, -1);
      else
        // stack: msg msg
        lua_remove(L, -2);
      // restore hook
      lua_sethook(L, hf, hm, hc);
    }
  // stack: msg
  return 1;
}



/*! A safe error handler that enrichs
  the error message with a traceback. */

int
luaQ_traceback(struct lua_State *L)
{
  return luaQ_tracebackskip(L, 0);
}




// ========================================
// Completion



/*! Returns potential completion for a string.
  Takes as input a single string composed of 
  identifiers separated by '.' or ':' and returns a table 
  representing an array of potential completions.
  Each completion is a string that could be reasonably 
  appended to the initial argument.
  This function throws errors when something goes wrong.
  Use \a pcall to call it safely. */

int 
luaQ_complete(struct lua_State *L)
{
  int k = 0;
  int loop = 0;
  const char *stem = luaL_checkstring(L, 1);
  lua_pushvalue(L, LUA_GLOBALSINDEX);
  for(;;)
    {
      const char *s = stem;
      while (*s && *s != '.' && *s != ':')
        s++;
      if (*s == 0)
        break;
      // stack: table str
      lua_pushlstring(L, stem, s-stem);
      lua_gettable(L, -2);
      // stack: ntable table str
      lua_replace(L, -2);
      // stack: ntable str
      stem = s + 1;
    }
  lua_createtable(L, 0, 0);
  lua_insert(L, -2);
  // stack: maybetable anstable str
  if (lua_isuserdata(L, -1) && lua_getmetatable(L, -1))
    {
      lua_replace(L, -2);
      lua_pushliteral(L, "__index");
      lua_rawget(L, -2);
      if (lua_isfunction(L, -1))
        {
          lua_pop(L, 1);
          lua_pushliteral(L, "__metatable");
          lua_rawget(L, -2);
        }
      lua_replace(L, -2);
    }
  if (! lua_istable(L, -1))
    {
      lua_pop(L, 1);
      return 1;
    }
  // stack: table anstable str
  size_t stemlen = strlen(stem);
  for(;;)
    {
      lua_pushnil(L);
      while (lua_next(L, -2))
        {
          // stack: value key table anstable str
          bool ok = false;
          size_t keylen;
          const char *key = lua_tolstring(L, -2, &keylen);
          if (key && keylen > 0 && keylen >= stemlen)
            if (!strncmp(key, stem, stemlen))
              ok = true;
          if (ok && stemlen==0 && !isalpha(key[0]))
            ok = false;
          if (ok)
            for (int i=0; ok && i<(int)keylen; i++)
              if (!isalpha(key[i]) && !isdigit(key[i]) && key[i]!='_')
                ok = false;
          if (ok)
            {
              const char *suffix = "";
              switch (lua_type(L, -1)) 
                {
                case LUA_TFUNCTION: 
                  suffix = "("; 
                  break;
                case LUA_TTABLE: 
                  suffix = ".";
                  luaQ_getfield(L, -1, "_C");
                  if (lua_istable(L, -1))
                    suffix = ":";
                  lua_pop(L, 1);
                  break;
                case LUA_TUSERDATA: {
                  QVariant v = luaQ_toqvariant(L, -1);
                  const char *s = QMetaType::typeName(v.userType());
                  if (s && !strcmp(s,"QtLuaMethodInfo"))
                    suffix = "(";
                  else if (s && !strcmp(s, "QtLuaPropertyInfo"))
                    suffix = "";
                  else if (lua_getmetatable(L, -1)) {
                    lua_pop(L, 1);
                    suffix = ":";
                  } else 
                    suffix = "";
                } break;
                default:
                  break;
                }
              // stack: value key table anstable str
              lua_pushfstring(L, "%s%s", key+stemlen, suffix);
              lua_rawseti(L, -5, ++k);
            }
          lua_pop(L, 1);
        }
      // stack: table anstable str
      if (! lua_getmetatable(L, -1))
        break;
      lua_replace(L, -2);
      lua_pushliteral(L, "__index");
      lua_rawget(L, -2);
      if (lua_isfunction(L, -1))
        {
          lua_pop(L, 1);
          lua_pushliteral(L, "__metatable");
          lua_rawget(L, -2);
        }
      lua_replace(L, -2);
      if (! lua_istable(L, -1))
        break;
      if (++loop > 100)
        luaL_error(L, "complete: infinite loop in metatables");
    }
  // stack: something anstable str
  lua_pop(L, 1);
  lua_replace(L, -2);
  return 1;
}




// ========================================
// Printing



static int
simple_print(lua_State *L)
{
  int i;
  int nr = lua_gettop(L);
  for (i=1; i<=nr; i++)
    {
      const char *s = lua_tostring(L, i);
      printf("%s", s ? s : "???");
      printf("%c", i<nr ? '\t' : '\n');
    }
  return 0;
}


int
luaQ_print(lua_State *L, int nr)
{
  int base = lua_gettop(L);
  nr = qMin(lua_gettop(L), nr);
  if (nr <= 0)
    return 0; 
  lua_getglobal(L, "print");
  if (lua_type(L, -1) != LUA_TFUNCTION)
    {
      lua_pop(L, 1);
      lua_pushcfunction(L, simple_print);
    }
  lua_checkstack(L, nr);
  for (int i=base-nr+1; i<=base; i++)
    lua_pushvalue(L, i);
  if (lua_pcall(L, nr, 0, 0))
    {
      const char *err = "error object is not a string";
      if (lua_isstring(L, -1))
        err = lua_tostring(L, -1);
      printf("error calling 'print' (%s)\n", err);
      lua_pop(L, 1);
    }
  return nr;
}




// ========================================
// Cross-thread calls



int 
luaQ_pcall(lua_State *L, int na, int nr, int eh, int oh)
{
  QtLuaEngine *engine = luaQ_engine(L);
  QObject *obj = engine;
  if (oh)
    obj = luaQ_toqobject(L, oh);
  if (! obj)
    luaL_error(L, "invalid qobject");
  return luaQ_pcall(L, na, nr, eh, obj);
}




// ========================================
// Qt library functions



static int
qt_connect(lua_State *L)
{
  // LUA: "qt.connect(object signal closure)" 
  // Connects signal to closure.
  
  // LUA: "qt.connect(object signal object signal_or_slot)"
  // Connects signal to signal or slot.
  
  QObject *obj = luaQ_checkqobject<QObject>(L, 1);
  const char *sig = luaL_checkstring(L, 2);
  QObject *robj = luaQ_toqobject(L, 3);
  if (robj)
    {
      // search signal or slot
      QByteArray rsig = luaL_checkstring(L, 4);
      const QMetaObject *mo = robj->metaObject();
      int idx = mo->indexOfMethod(rsig.constData());
      if (idx < 0)
        {
          rsig = QMetaObject::normalizedSignature(rsig.constData());
          idx = mo->indexOfMethod(rsig.constData());
          if (idx < 0)
            luaL_error(L, "cannot find target slot or signal %s", 
                       rsig.constData());
        }
      // prepend signal or slot indicator
      QMetaMethod method = mo->method(idx);
      if (method.methodType() == QMetaMethod::Signal)
        rsig.prepend('0' + QSIGNAL_CODE);
      else if (method.methodType() == QMetaMethod::Slot)
        rsig.prepend('0' + QSLOT_CODE);
      else
        luaL_error(L, "target %s is not a slot or a signal",
                   rsig.constData());
      // connect
      QByteArray ssig = sig;
      ssig.prepend('0' + QSIGNAL_CODE);
      if (! QObject::connect(obj, ssig.constData(), robj, rsig.constData()))
        luaL_error(L, "cannot find source signal %s", sig);
    }
  else
    {
      luaL_checktype(L, 3, LUA_TFUNCTION);
      bool direct = lua_toboolean(L, 4);
      if (direct)
        luaL_checktype(L, 4, LUA_TBOOLEAN);
      if (! luaQ_connect(L, obj, sig, 3, direct))
        luaL_error(L, "cannot find source signal %s", sig);
    }
  return 0;
}


static int
qt_disconnect(lua_State *L)
{
  // LUA: qt.disconnect(object [signal [closure]])
  // LUA: qt.disconnect(object signal object signal_or_slot)
  // Disconnect all connections between 
  // the specified signal and a lua function.
  // Returns boolean indicating if such signal were found.
  bool ok = false;
  QObject *obj = luaQ_checkqobject<QObject>(L, 1);
  const char *sig = luaL_optstring(L, 2, 0);
  int narg = lua_gettop(L);
  if (narg>3 || lua_isuserdata(L, 3))
    {
      QObject *robj = luaQ_optqobject<QObject>(L, 3, 0);
      const char *rsig = luaL_optstring(L, 4, 0);
      QByteArray bsig(sig);
      QByteArray brsig(rsig);
      bsig.prepend('0'+QSIGNAL_CODE);
      brsig.prepend('0'+QSLOT_CODE);
      sig = (sig) ? bsig.constData() : 0;
      rsig = (rsig) ? brsig.constData() : 0;
      ok = QObject::disconnect(obj, sig, robj, rsig);
      brsig[0] = '0' + QSIGNAL_CODE;
      if (rsig)
        ok |= QObject::disconnect(obj, sig, robj, brsig.constData());
    }
  else
    {
      int findex = 3;
      if (lua_isnoneornil(L, 3))
        findex = 0;
      else if (! lua_isfunction(L, 3))
        luaL_typerror(L, 3, "function");
      ok = luaQ_disconnect(L, obj, sig, findex);
    }
  lua_pushboolean(L, ok);
  return 1;
}


static int
qt_doevents(lua_State *L)
{
  luaQ_doevents(L, lua_toboolean(L, 1));
  return 0;
}


static int
qt_qcall(lua_State *L)
{
  QObject *obj = luaQ_checkqobject<QObject>(L, 1);
  luaL_checktype(L, 2, LUA_TFUNCTION);
  int base = 2;
  int narg = lua_gettop(L) - base;
  int status = luaQ_pcall(L, narg, LUA_MULTRET, 0, obj);
  lua_pushboolean(L, !status);
  lua_insert(L, base);
  return lua_gettop(L) - base + 1;
}


static int
qt_xqcall(lua_State *L)
{
  QObject *obj = luaQ_checkqobject<QObject>(L, 1);
  lua_insert(L, 2);
  luaL_checktype(L, 2, LUA_TFUNCTION);
  luaL_checktype(L, 3, LUA_TFUNCTION);
  int base = 3;
  int narg = lua_gettop(L) - base;
  int status = luaQ_pcall(L, narg, LUA_MULTRET, 2, obj);
  lua_pushboolean(L, !status);
  lua_insert(L, base);
  return lua_gettop(L) - base + 1;
}


static int
qt_type(lua_State *L)
{
  QVariant v;
  if (lua_type(L, 1) == LUA_TUSERDATA)
    v = luaQ_toqvariant(L, 1);
  if (v.type() != QVariant::Invalid)
    {
      lua_pushvalue(L, 1);
      lua_getfield(L, -1, "type");
      if (lua_isfunction(L, -1))
        {
          lua_insert(L, -2);
          lua_call(L, 1, 1);
          return 1;
        }
    }
  lua_pushnil(L);
  return 1;
}


static int
qt_isa(lua_State *L)
{
  QVariant v;
  if (lua_type(L, 1) == LUA_TUSERDATA)
    v = luaQ_toqvariant(L, 1);
  if (v.type() != QVariant::Invalid)
    {
      lua_pushvalue(L, 1);
      lua_getfield(L, -1, "isa");
      if (lua_isfunction(L, -1))
        {
          lua_insert(L, 1);
          lua_call(L, lua_gettop(L)-1, 1);
          return 1;
        }
    }
  lua_pushnil(L);
  return 1;
}



// {{{ functions copied or derived from loadlib.c

static int readable (const char *filename) 
{  
  FILE *f = fopen(filename, "r");  /* try to open file */
  if (f == NULL) return 0;  /* open failed */
  fclose(f);
  return 1;
}

static const char *pushnexttemplate (lua_State *L, const char *path) 
{
  const char *l;
  while (*path == *LUA_PATHSEP) path++;  /* skip separators */
  if (*path == '\0') return NULL;  /* no more templates */
  l = strchr(path, *LUA_PATHSEP);  /* find next separator */
  if (l == NULL) l = path + strlen(path);
  lua_pushlstring(L, path, l - path);  /* template */
  return l;
}

static const char *pushfilename (lua_State *L, const char *name) 
{
  const char *path;
  const char *filename;
  luaQ_getfield(L, LUA_GLOBALSINDEX, "package");
  luaQ_getfield(L, -1, "cpath");
  lua_remove(L, -2);
  if (! (path = lua_tostring(L, -1)))
    luaL_error(L, LUA_QL("package.cpath") " must be a string");
  lua_pushliteral(L, ""); 
  while ((path = pushnexttemplate(L, path))) {
    filename = luaL_gsub(L, lua_tostring(L, -1), "?", name);
    lua_remove(L, -2);
    if (readable(filename))
      { // stack:  cpath errmsg filename
        lua_remove(L, -3);
        lua_remove(L, -2);
        return lua_tostring(L, -1);
      }
    lua_pushfstring(L, "\n\tno file " LUA_QS, filename);
    lua_remove(L, -2); /* remove file name */
    lua_concat(L, 2);  /* add entry to possible error message */
  }
  lua_pushfstring(L, "module " LUA_QS " not found", name);
  lua_replace(L, -3);
  lua_concat(L, 2);
  lua_error(L);
  return 0;
}

// functions copied or derived from loadlib.c }}}

static int
qt_require(lua_State *L)
{
  const char *name = luaL_checkstring(L, 1);
  lua_settop(L, 1);
  lua_getfield(L, LUA_REGISTRYINDEX, "_LOADED");  // index 2
  lua_getfield(L, 2, name);
  if (lua_toboolean(L, -1))
    return 1;
  const char *filename = pushfilename(L, name);  // index 3
  QLibrary library(QString::fromLocal8Bit(filename));
  library.setLoadHints(QLibrary::ExportExternalSymbolsHint);
  if (! library.load())
    luaL_error(L, "cannot load " LUA_QS, filename);
  lua_pushfstring(L, "luaopen_%s", name);  // index 4
  lua_CFunction func = (lua_CFunction)library.resolve(lua_tostring(L, -1));
  if (! func)
    luaL_error(L, "no symbol " LUA_QS " in module " LUA_QS, 
               lua_tostring(L, -1), filename);
  lua_pushboolean(L, 1);
  lua_setfield(L, 2, name);
  lua_pushcfunction(L, func);
  lua_pushstring(L, name);
  lua_call(L, 1, 1);
  if (! lua_isnil(L, -1))
    lua_setfield(L, 2, name);
  lua_getfield(L, 2, name);
  return 1;
}


static int
qt_pause(lua_State *L)
{
  luaQ_pause(L);
  return 0;
}

static int
qt_resume(lua_State *L)
{
  luaQ_resume(L, lua_toboolean(L, 1));
  return 0;
}


static const luaL_Reg qt_lib[] = {
  {"connect", qt_connect},
  {"disconnect", qt_disconnect},
  {"doevents", qt_doevents},
  {"qcall", qt_qcall},
  {"xqcall", qt_xqcall},
  {"type", qt_type},
  {"typename", qt_type},
  {"isa", qt_isa},
  {"require", qt_require},
  {"pause", qt_pause},
  {"resume", qt_resume},
  {0, 0}
};


static int
qt_m__index(lua_State *L)
{
  const char *s = luaL_checkstring(L, 2);
  QtLuaEngine *engine = luaQ_engine(L);
  QObject *obj = engine->namedObject(s);
  if (obj)
    luaQ_pushqt(L, obj);
  else
    lua_pushnil(L);
  return 1;
}


// ========================================
// Initialize

static int
no_methodcall(lua_State *L)
{
  luaL_error(L, "This class prevents lua to call this method");
  return 0;
}

static void
hide_deletelater(lua_State *L, const QMetaObject *mo)
{
  luaQ_pushmeta(L, mo);
  lua_getfield(L, -1, "__metatable");
  // ..stack: metaclass
  lua_pushcfunction(L, no_methodcall);
  lua_setfield(L, -2, "deleteLater");
  lua_pushcfunction(L, no_methodcall);
  lua_setfield(L, -2, "deleteLater(QObject*)");
  // restore
  lua_pop(L, 1);
}


int  
luaopen_qt(lua_State *L)
{
  const char *qt = luaL_optstring(L, 1, "qt");
  luaQ_pushqt(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, qt);
  luaL_register(L, qt, qt_lib);

  // Add qt_m_index in a metatable
  lua_createtable(L, 0, 1);
  lua_pushcfunction(L, qt_m__index);
  lua_setfield(L, -2, "__index");
  lua_setmetatable(L, -2);

  // Hide deleteLater in selected classes
  hide_deletelater(L, &QCoreApplication::staticMetaObject);
  hide_deletelater(L, &QtLuaEngine::staticMetaObject);

  return 1;
}





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


