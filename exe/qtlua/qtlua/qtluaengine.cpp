// -*- C++ -*-

#define QTLUAENGINE 1

#include "qtluautils.h"
#include "qtluaengine.h"
#include "lualib.h"

#include <QCoreApplication>
#include <QDebug>
#include <QEvent>
#include <QEventLoop>
#include <QFile>
#include <QList>
#include <QMap>
#include <QMetaMethod>
#include <QMetaObject>
#include <QMetaProperty>
#include <QMetaType>
#include <QMutex>
#include <QMutexLocker>
#include <QPointer>
#include <QSet>
#include <QTimer>
#include <QThread>
#include <QVector>
#include <QWaitCondition>

#include <limits.h>



// ========================================
// Declaration

static int luaQ_pcall(lua_State *L, 
                      int na, int nr, int eh, 
                      QObject *obj, bool async);



/* miscellaneous information */

typedef QVector<int> IntVector;
typedef QVector<void*> PtrVector;
typedef QVector<QVariant> VarVector;

struct QtLuaMethodInfo
{
  struct Detail { 
    int id; 
    PtrVector types;
  };
  const QMetaObject *metaObject;
  QVector<Detail> d;
};

Q_DECLARE_METATYPE(QtLuaMethodInfo)
  
struct QtLuaPropertyInfo
{
  int id;
  const QMetaObject *metaObject;
  QMetaProperty metaProperty;
};

Q_DECLARE_METATYPE(QtLuaPropertyInfo)

Q_DECLARE_METATYPE(QVariant)



// ========================================
// Registry keys


static const char *engineKey = "engine";
static const char *signalKey = "signals";
static const char *objectKey = "objects";
static const char *metaKey = "metatables";
static const char *qtKey = "qt";


// Summary of registry entries:
// _R[engineKey] : pointer to QtLuaEngine::Private.
// _R[qtKey] : qt package.
// _R[qtKey].typename : class for metatype "typename".
// _R[qtKey].classname : class for metaobject "classname".
// _R[metaKey][typeid] : metatable for metatype (by typeid) 
// _R[metaKey][metaobjectptr] : metatable for metaobject (by ptr)
// _R[objectKey][objectptr] : lua value for qobject (weak table)
// _R[signalKey][receiverptr] : closure for signal receiver.
// _R[metatable] : equal to qtKey if this is a qt object.


static void
luaQ_setup(lua_State *L, QtLuaEngine::Private *d)
{
  // metatypes
  qRegisterMetaType<QVariant>("QVariant");
  qRegisterMetaType<QtLuaMethodInfo>("QtLuaMethodInfo");
  qRegisterMetaType<QtLuaPropertyInfo>("QtLuaPropertyInfo");
  qRegisterMetaType<QObjectPointer>("QObjectPointer");
  // engine
  lua_pushlightuserdata(L, (void*)engineKey);
  lua_pushlightuserdata(L, (void*)d);
  lua_rawset(L, LUA_REGISTRYINDEX);
  // metatables
  lua_pushlightuserdata(L, (void*)metaKey);
  lua_createtable(L, 0, 0);
  lua_rawset(L, LUA_REGISTRYINDEX);
  // signals
  lua_pushlightuserdata(L, (void*)signalKey);
  lua_createtable(L, 0, 0);
  lua_rawset(L, LUA_REGISTRYINDEX);
  // objects [weak table]
  lua_pushlightuserdata(L, (void*)objectKey);
  lua_createtable(L, 0, 0); 
  lua_createtable(L, 0, 1); 
  lua_pushliteral(L, "v");
  lua_setfield(L, -2, "__mode");
  lua_setmetatable(L, -2);
  lua_rawset(L, LUA_REGISTRYINDEX);
  // qt
  lua_pushlightuserdata(L, (void*)qtKey);
  lua_createtable(L, 0, 0);
  lua_rawset(L, LUA_REGISTRYINDEX);
  // package.preload["qt"]
  lua_getfield(L, LUA_GLOBALSINDEX, "package");
  if (lua_istable(L, -1)) 
    {
      lua_getfield(L, -1, "preload");
      if (lua_istable(L, -1))
        {
          lua_pushcfunction(L, luaopen_qt);
          lua_setfield(L, -2, "qt");
        }
      lua_pop(L, 1);
    }
  lua_pop(L, 1);
}


static QtLuaEngine::Private *
luaQ_private_noerr(lua_State *L)
{
  QtLuaEngine::Private *d = 0;
  lua_pushlightuserdata(L, (void*)engineKey);
  lua_rawget(L, LUA_REGISTRYINDEX);
  if (lua_islightuserdata(L, -1))
    d = static_cast<QtLuaEngine::Private*>(lua_touserdata(L, -1));
  lua_pop(L, 1);
  return d;
}


static QtLuaEngine::Private *
luaQ_private(lua_State *L)
{
  QtLuaEngine::Private *d = luaQ_private_noerr(L);
  if (! d)
    luaL_error(L, "qtlua: not running inside a QtLuaEngine.");
  return d;
}




// ========================================
// QtLuaEngine::Global

struct QtLuaEngine::Global {
  QMutex mutex;
  QMap<QByteArray,const QMetaObject*> knownMetaObjects;
  void registerMetaObject(const QMetaObject *mo, bool super=true);
  const QMetaObject *findMetaObject(QByteArray className);
};


Q_GLOBAL_STATIC(QtLuaEngine::Global, qtLuaEngineGlobal);


void 
QtLuaEngine::Global::registerMetaObject(const QMetaObject *mo, bool super)
{
  QMutexLocker locker(&mutex);
  knownMetaObjects[mo->className()] = mo;
  while (super && (mo = mo->superClass()))
    knownMetaObjects[mo->className()] = mo;
}


const QMetaObject *
QtLuaEngine::Global::findMetaObject(QByteArray className)
{
  QMutexLocker locker(&mutex);
  if (knownMetaObjects.contains(className))
    return knownMetaObjects[className];
  return 0;
}



// ========================================
// QtLuaEngine::Private basics


struct QtLuaQueuedSignal {
  QPointer<QObject> sender;
  QPointer<QtLuaEngine::Receiver> receiver;
  void *delsignal;
  VarVector args;
};


struct QtLuaEngine::Protector  : public QObject
{
  Q_OBJECT
  Private *d;
  QVariantList saved;
  QMutex mutex;
public:
  Protector(QtLuaEngine::Private *d) : d(d) { }
  bool maybeProtect(const QVariant &var);
  bool protect(const QVariant &var);
  bool event(QEvent *e);
};


struct QtLuaEngine::Private  : public QObject
{
  Q_OBJECT
public:
  Private(QtLuaEngine *parent);
  ~Private();
  QThread *luaThread() const;
  bool isObjectLuaOwned(QObject *obj);
  void makeObjectLuaOwned(QObject *obj);
  bool processQueuedSignals(QMutexLocker &locker);
  void disconnectAllSignals();
  bool stopHelper(bool unwind);
  bool resumeHelper(int retcode);
  static void stopHook(lua_State *L, lua_Debug *ar);
public slots:
  void objectDestroyed(QObject*);
  void readySlot();
  void queueSlot();
  void stopSlot() {}
  void emitStateChanged(State s) { emit stateChanged(s); }
  void emitPauseSignal() { emit stateChanged(QtLuaEngine::Paused); }
  void emitReadySignal() { emit readySignal(); }
  void emitQueueSignal() { emit queueSignal(); }
  void emitErrorMessage(QByteArray m) { emit errorMessage(m); }
  void emitStopSignal() { emit stopSignal(); }
signals:
  void errorMessage(QByteArray message);
  void stateChanged(int s);
  void readySignal();
  void queueSignal();
  void stopSignal();
public:
  QtLuaEngine *q;
  lua_State   *L;
  // locking
  QMutex mutex;
  QWaitCondition condition;
  int lockCount;
  QThread *lockThread;
  bool rflag;
  // hopping
  QWaitCondition hopCondition;
  QEventLoop *hopLoop;
  QEvent *hopEvent;
  int hopNA;
  int hopNR;
  int hopEH;
  // pausing
  QEventLoop *pauseLoop;
  QByteArray lastErrorMessage;
  QList<QByteArray> lastErrorLocation;
  bool printResults;
  bool printErrors;
  bool pauseOnError;
  bool unwindStack;
  bool resumeFlag;
  bool errorHandlerFlag;
  // debugging
  lua_Debug *hookInfo;
  lua_Hook hookFunction;
  int hookMask;
  int hookCount;
  // protector
  Protector *protector;
  // maps
  QMap<QByteArray,const QMetaObject*> knownMetaObjects;
  QMap<QString,QObjectPointer> namedObjectsCache;
  QSet<QObject*> namedObjects;
  QSet<QObject*> luaOwnedObjects;
  QList<QtLuaQueuedSignal> queuedSignals;
};


QtLuaEngine::Private::Private(QtLuaEngine *parent)
  : QObject(parent),
    q(parent),
    L(0),
    lockCount(0),
    lockThread(0),
    rflag(true),
    hopLoop(0),
    hopEvent(0),
    pauseLoop(0),
    printResults(false),
    printErrors(true),
    pauseOnError(false),
    unwindStack(false),
    resumeFlag(false),
    errorHandlerFlag(false),
    hookInfo(0),
    hookFunction(0),
    hookMask(0),
    hookCount(0),
    protector(0)
{
  // setup paths
#if HAVE_LUA_EXECUTABLE_DIR
  QString path = QCoreApplication::applicationFilePath();
  lua_executable_dir(QFile::encodeName(path).constData());
#endif
  // create lua interpreter
  L = luaL_newstate();
  Q_ASSERT(L);
  luaL_openlibs(L);
  luaQ_setup(L, this);
  // delayed connections
  connect(this, SIGNAL(readySignal()), 
          this, SLOT(readySlot()),
          Qt::QueuedConnection );
  connect(this, SIGNAL(queueSignal()), 
          this, SLOT(queueSlot()),
          Qt::QueuedConnection );
  connect(this, SIGNAL(stopSignal()), 
          this, SLOT(stopSlot()),
          Qt::QueuedConnection );
  connect(this, SIGNAL(stateChanged(int)), 
          q, SIGNAL(stateChanged(int)));
  connect(this, SIGNAL(errorMessage(QByteArray)), 
          q, SIGNAL(errorMessage(QByteArray)));
}


QtLuaEngine::Private::~Private()
{
  // close interpreter
  lua_pushlightuserdata(L, (void*)engineKey);
  lua_pushnil(L);
  lua_rawset(L, LUA_REGISTRYINDEX);
  lua_close(L);
  L = 0;
  // destroy protector
  if (protector)
    protector->deleteLater();
  protector = 0;
}


QThread *
QtLuaEngine::Private::luaThread() const
{
  if (lockThread)
    return lockThread;
  return thread();
}


void
QtLuaEngine::Private::objectDestroyed(QObject *obj)
{
  QMutexLocker locker(&mutex);
  if (luaOwnedObjects.contains(obj))
    luaOwnedObjects.remove(obj);
  if (namedObjects.contains(obj))
    namedObjects.remove(obj);
}


bool 
QtLuaEngine::Private::isObjectLuaOwned(QObject *obj)
{
  QMutexLocker locker(&mutex);
  return luaOwnedObjects.contains(obj);
}


void 
QtLuaEngine::Private::makeObjectLuaOwned(QObject *obj)
{
  if (obj)
    {
      connect(obj, SIGNAL(destroyed(QObject*)),
              this, SLOT(objectDestroyed(QObject*)),
              Qt::DirectConnection );
      QMutexLocker locker(&mutex);
      luaOwnedObjects += obj;
    }
}


static void
queue_sub(QtLuaEngine::Private *d, QMutexLocker &locker)
{
  QThread *mythread = QThread::currentThread();
  Q_ASSERT(mythread == d->thread());
  Q_ASSERT(!d->lockCount && !d->hopEvent && !d->hopLoop);
  d->lockCount = 1;
  d->lockThread = mythread;
  d->processQueuedSignals(locker); 
  if (! d->queuedSignals.isEmpty())
    d->emitQueueSignal();
  d->lockCount = 0;
  d->lockThread = 0;
  d->condition.wakeOne();
}


void 
QtLuaEngine::Private::readySlot()
{
  QMutexLocker locker(&mutex);
  if (!lockCount && !hopLoop && !hopEvent && !pauseLoop)
    queue_sub(this, locker);
  bool stateChanged = rflag;
  hookInfo = 0;
  unwindStack = false;
  rflag = false;
  if (lua_gethook(L) == stopHook)
    lua_sethook(L, hookFunction, hookMask, hookCount);
  locker.unlock();
  if (stateChanged)
    emit q->stateChanged(QtLuaEngine::Ready);
}


void 
QtLuaEngine::Private::queueSlot()
{
  QMutexLocker locker(&mutex);
  if (pauseLoop && ! hookInfo)
    {
      Q_ASSERT(rflag);
      QEventLoop *savedLoop = pauseLoop; 
      lua_Debug *savedInfo = hookInfo;
      pauseLoop = 0;
      hookInfo = 0;
      rflag = false;
      resumeFlag = false;
      queue_sub(this, locker);
      bool stateChanged = rflag;
      pauseLoop = savedLoop;
      hookInfo = savedInfo;
      rflag = true;
      locker.unlock();
      if (stateChanged)
        emit q->stateChanged(QtLuaEngine::Paused);
      if (pauseLoop && unwindStack)
        pauseLoop->exit(1);
      else if (pauseLoop && resumeFlag)
        pauseLoop->exit(0);
    }
  else if (!lockCount && !lockThread && !hopEvent && !pauseLoop)
    {
      queue_sub(this, locker);
      bool stateChanged = rflag;
      hookInfo = 0;
      unwindStack = false;
      rflag = false;
      if (lua_gethook(L) == stopHook)
        lua_sethook(L, hookFunction, hookMask, hookCount);
      locker.unlock();
      if (stateChanged)
        emit q->stateChanged(QtLuaEngine::Ready);    
    }
}




// ========================================
// QtLuaEngine state



/*! Returns the engine state. */

QtLuaEngine::State 
QtLuaEngine::state() const
{
  QMutexLocker locker(&d->mutex);
  if (d->pauseLoop)
    return Paused;
  else if (d->rflag)
    return Running;
  else
    return Ready;
}

/*! \fn QtLuaEngine::isReady()
  Returns \a true if the engine is in ready state. */

/*! \fn QtLuaEngine::isRunning()
  Returns \a true if the engine is in running state. */

/*! \fn QtLuaEngine::isPaused()
  Returns \a true if the engine is in paused state. */


/*! Returns true if the engine is paused because an error has occured */

bool
QtLuaEngine::isPausedOnError() const
{
  QMutexLocker locker(&d->mutex);
  if (d->pauseLoop && d->errorHandlerFlag)
    return true;
  return false;
}


/*! \signal stateChanged(State state)
  Posted when the engine state changes.
  When this message is received, 
  the engine state might have changed again! */


/*! \signal errorMessage(QByteArray message)
  Posted when a lua error is signalled. */



// ========================================
// QtLuaLocker


/*! \class QtLuaLocker
  This class safely locks a \a QtLuaEngine.
  The constructor acquires the lock 
  and the destructor releases the lock.
  The \a lua_State can then be safely 
  accessed by casting an instance of this class. */


QtLuaLocker::~QtLuaLocker()
{
  if (engine && count)
    {
      QtLuaEngine::Private *d = engine->d;
      d->mutex.lock();
      count = d->lockCount - 1;
      if (! count)
        {
          if (d->hopEvent)
            {
              d->hopCondition.wakeOne(); // mostly for async eval
            }
          else
            {
              d->lockThread = 0;
              d->emitReadySignal();
              d->condition.wakeOne();
            }
        }
      d->lockCount = count;
      d->mutex.unlock();
    }
}


/*! Constructor. 
  This constructor acquires a lock,
  waiting as long as necessary. */

QtLuaLocker::QtLuaLocker(QtLuaEngine *engine)
  : engine(engine), count(0)
{
  QtLuaEngine::Private *d = engine->d;
  QThread *mythread = QThread::currentThread();
  QMutexLocker locker(&d->mutex);
  for(;;)
    {
      if (d->lockCount > 0 && d->lockThread == mythread)
        break;
      if (d->lockCount == 0 && d->lockThread == 0)
        break;
      d->condition.wait(&d->mutex);
    }
  d->lockCount += 1;
  d->lockThread = mythread;
  count = d->lockCount;
}



/*! Alternate constructor. 
  This constructor attempts to acquire a lock
  waiting at most \a timeOut milliseconds.
  To see if the lock has been acquired, 
  use the \a lua_State* conversion operator
  or use the \a isReady() function. */

QtLuaLocker::QtLuaLocker(QtLuaEngine *engine, int timeOut)
  : engine(engine), count(0)
{
  QtLuaEngine::Private *d = engine->d;
  QThread *mythread = QThread::currentThread();
  QMutexLocker locker(&d->mutex);
  for(;;)
    {
      if (d->lockCount > 0 && d->lockThread == mythread)
        break;
      if (d->lockCount == 0 && d->lockThread == 0)
        break;
      if (! d->condition.wait(&d->mutex, timeOut))
        return;
    }
  d->lockCount += 1;
  d->lockThread = mythread;
  count = d->lockCount;
}


/*! \fn bool isReady()
  Returns true if the locking operation was 
  successful and the interpreter is in ready state. 
  Note that locking and state are distinct concepts.
  It is possible to lock a running interpreter
  while it is waiting for other events.
  The \a eval and \a evaluate functions use this test
  to decide whether to run a command. */


/*! \fn operator lua_State*
  Accesses the \a lua_State underlying the locked engine.
  This provides the only safe way to access 
  the \a lua_State variable. */


/*! Calling this function after a successful lock 
  causes the engine to transition to state \a Running.
  The engine will return to state \a Ready after
  the destruction of the last \a QtLuaLocker object 
  and the execution of the command queue.
  Temporary releasing the lock with \a unlock() 
  keeps the engine in state \a Running. */

void 
QtLuaLocker::setRunning()
{
  bool stateChanged = false;
  QtLuaEngine::Private *d = engine->d;
  if (count > 0)
    {
      QMutexLocker locker(&d->mutex);
      stateChanged = !d->rflag;
      d->rflag = true;
    }
  if (stateChanged)
    d->emitStateChanged(QtLuaEngine::Running);
}







// ========================================
// QtLuaEngine basics


/*! \class QtLuaEngine
  Class \a QtLuaEngine represents a Lua interpreter.
  This object can be used to add a Lua interpreter 
  to any Qt application with capabilities 
  comparable to those of the QtScript language
  and additional support for multi-threaded execution.

  Instances of this class can be in one of three state.
  State \a QtLuaEngine::Ready indicates that the interpreter
  is ready to accept new Lua commands.
  State \a QtLuaEngine::Running indicates that the interpreter
  is currently executing a Lua program.
  State \a QtLuaEngine::Paused indicates that the interpreter
  was suspended while executing a Lua program. One can then
  use the Lua debug library to investigage the Lua state.

  Class \a QtLuaEngine provides member functions to 
  submit Lua strings to the interpreter and to collect 
  the evaluation results.  If these functions are invoked
  from the thread owning the Lua engine object,
  the Lua code is executed right away. 
  Otherwise a thread hopping operation with \a luaQ_pcall
  ensures that the execution of a Lua program happens 
  in the thread owning the Lua engine object. */


QtLuaEngine::~QtLuaEngine()
{
  Q_ASSERT(QThread::currentThread() == thread());
  // make as silent as possible
  setPrintResults(false);
  setPauseOnError(false);
  disconnect(d, 0, this, 0);
  d->disconnectAllSignals();
  // stop lua
  QMutexLocker locker(&d->mutex);
  while (d->lockCount || d->lockThread || d->hopEvent)
    {
      locker.unlock();
      resume(true);
      stop(true);
      QEventLoop loop;
      loop.processEvents(QEventLoop::ExcludeUserInputEvents, 100);
      locker.relock();
      d->unwindStack = true;
    }
  // delete children and owned objects
  QObject *obj;
  QList<QObjectPointer> family;
  foreach(obj, d->luaOwnedObjects)
    family << obj;
  foreach(obj, children())
    family << obj;
  d->luaOwnedObjects.clear();
  foreach(QObjectPointer objptr, family)
    if ((obj = objptr) && (obj != d))
      obj->deleteLater();
  // disconnect protector
  if (d->protector)
    QCoreApplication::instance()->removeEventFilter(d->protector);
}


/*! Basic constructor. */

QtLuaEngine::QtLuaEngine(QObject *parent)
  : QObject(parent),
    d(new Private(this)),
    L(d->L)
{
  d->protector = new QtLuaEngine::Protector(d);
  d->protector->moveToThread(QCoreApplication::instance()->thread());
  QCoreApplication::instance()->installEventFilter(d->protector);
}


/*! Registering a metaobject allows the engine 
  to recognize qobject classes by name in method 
  and signal arguments. This happens automatically 
  when a qobject is translated into a lua userdata,
  or, more generally, whenever \a luaQ_pushmeta is called.
  In rare occasions, it may be necessary to manually 
  register a meta object, for instance when a scriptable 
  method returns a qobject whose class has not been previously
  registered and is not a superclass of the current class,
  or when one connects a signal whose arguments are
  qobjects whose class has not been previously registered. */

void 
QtLuaEngine::registerMetaObject(const QMetaObject *mo)
{
  qtLuaEngineGlobal()->registerMetaObject(mo);
}


/*! Make a \a QObject accessible to the interpreter by name.
  Use the Qt object name when string \a name is not provided. 
  The lua engine keeps tracking the object when its name
  is reset or changed. */

void 
QtLuaEngine::nameObject(QObject *object, QString name)
{
  if (object)
    {
      if (! name.isEmpty())
        object->setObjectName(name);
      name = object->objectName();
      QMutexLocker locker(&d->mutex);
      d->namedObjects += object;
      if (! name.isEmpty())
        d->namedObjectsCache[name] = object;
      connect(object, SIGNAL(destroyed(QObject*)),
              d, SLOT(objectDestroyed(QObject*)),
              Qt::DirectConnection );
    }
}


/*! Returns a \a QObject by name. */

QObject *
QtLuaEngine::namedObject(QString name)
{
  QObject *obj;
  QMutexLocker locker(&d->mutex);
  if (d->namedObjectsCache.contains(name))
    {
      obj = d->namedObjectsCache[name];
      if (obj && obj->objectName() == name)
        return obj;
      d->namedObjectsCache.remove(name);
    }
  foreach(obj, d->namedObjects)
    if (obj->objectName() == name)
      {
        d->namedObjectsCache[name] = obj;
        return obj;
      }
  return 0;
}


/*! Return the list of all named objects. */

QList<QObjectPointer> 
QtLuaEngine::allNamedObjects() 
{
  QObject *obj;
  QList<QObjectPointer> list;
  foreach(obj, d->namedObjects)
    list += obj;
  return list;
}


/*! \property QtLuaEngine::lastErrorMessage
  Contains the last error message
  reported with signal \a errorMessage. */

QByteArray
QtLuaEngine::lastErrorMessage() const
{
  return d->lastErrorMessage;
}


/*! \property QtLuaEngine::lastErrorLocation
  Contains the location associated with the 
  last error message reported with signal \a errorMessage.
  When the location corresponds to a file, 
  this string has the format "@filename:linenumber". */

QStringList 
QtLuaEngine::lastErrorLocation() const
{
  QStringList list;
  foreach(QByteArray b, d->lastErrorLocation)
    list << QString::fromLocal8Bit(b);
  return list;
}




/*! \property QtLuaEngine::printResults
  Indicates whether the results of an \a eval()
  must be printed on the standard output. */

bool 
QtLuaEngine::printResults() const
{
  return d->printResults;
}


void 
QtLuaEngine::setPrintResults(bool b)
{
  d->printResults = b;
}


/*! \property QtLuaEngine::printErrors
  Indicates whether error messages
  must be printed on the standard output. */

bool 
QtLuaEngine::printErrors() const
{
  return d->printErrors;
}


void 
QtLuaEngine::setPrintErrors(bool b)
{
  d->printErrors = b;
}



/*! \property QtLuaEngine::pauseOnError
  Indicates whether the default error handler
  should pause execution when an error occurs
  as if \a pause() had been called from 
  the error handler. */

bool 
QtLuaEngine::pauseOnError() const
{
  return d->pauseOnError;
}


void 
QtLuaEngine::setPauseOnError(bool b)
{
  d->pauseOnError = b;
}


/*! \property QtLuaEngine::runSignalHandlers
  This property is true when the Lua interpreter
  honors the signal handler invokations immediately 
  instead of queuing them for further processing. */

bool 
QtLuaEngine::runSignalHandlers() const
{
  QMutexLocker locker(&d->mutex);
  if (! d->rflag)
    return true;
  if (d->pauseLoop && ! d->hookInfo)
    return true;
  return false;
}




// ========================================
// Stopping, Resuming, Evaluating


static bool
lua_pause_sub(lua_State *L, QtLuaEngine::Private *d, QMutexLocker &locker)
{
  // must be called with a message on the stack
  if (!d->pauseLoop && d->rflag && !d->unwindStack)
    {
      int savedLockCount = d->lockCount;
      QThread *savedLockThread = d->lockThread;
      QEventLoop loop;
      d->pauseLoop = &loop;
      d->lockCount = 0;
      d->lockThread = 0;
      d->condition.wakeOne();
      locker.unlock();
      // make sure we enter the loop before signalling.
      QTimer::singleShot(0, d, SLOT(emitPauseSignal()));
      if (loop.exec())
        d->unwindStack = true;
      locker.relock();
      d->lockThread = savedLockThread;
      d->lockCount = savedLockCount;
      d->pauseLoop = 0;
      return true;
    }
  return false;
}


void
QtLuaEngine::Private::stopHook(lua_State *L, lua_Debug *ar)
{
  QtLuaEngine::Private *d = luaQ_private_noerr(L);
  lua_sethook(L, 0, 0, 0);
  lua_pushliteral(L, "stop");
  luaQ_tracebackskip(L, 1);
  bool stateChanged = false;
  if (d) 
    {
      QMutexLocker locker(&d->mutex);
      lua_Debug *savedInfo = d->hookInfo;
      d->hookInfo = ar;
      lua_sethook(L, d->hookFunction, d->hookMask, d->hookCount);
      if (! d->unwindStack)
        stateChanged = lua_pause_sub(L, d, locker);
      d->hookInfo = savedInfo;
    }
  if (d && stateChanged)
    d->emitStateChanged(QtLuaEngine::Running);
  lua_pop(L, 1);
  if (d && !d->unwindStack)
    return;
  lua_pushliteral(L, "stop");
  lua_error(L);
}


static QList<QByteArray>
find_error_location(lua_State *L, QByteArray &message)
{
  QByteArray loc;
  QList<QByteArray> list;
  // parse locations in message
  const char *m = message.constData();
  for(const char *s = m; *s && *s != '\n'; s++)
    {
      // search ":[0-9]+: +"
      if (*s != ':')
        continue;
      char *e;
      QByteArray location(m, s-m);
      int lineno = (int)strtol(++s, &e, 10);
      if (lineno <= 0 || e <= s || *e != ':')
        continue;
      s = e;
      while (s[1] && s[1] == ' ')
        s += 1;
      m = s + 1;
      // append
      if (location.startsWith("[string \"") && location.endsWith("\"]"))
        location = location.mid(9, location.length()-11);
      else
        location = "@" + location;
      loc = location + ":" + QByteArray::number(lineno);
      if (list.isEmpty() || list.first() != loc)
        list.prepend(loc);
    }
  message = QByteArray(m);
  // parse locations in top ten stack elements
  int maxlevel = 8;
  int level = 0;
  lua_Debug ar;
  while (level < maxlevel && lua_getstack(L, level++, &ar)) 
    {
      lua_getinfo(L, "Snl", &ar);
      if (ar.currentline > 0)
        {
          loc = QByteArray(ar.source) + ":" + 
            QByteArray::number(ar.currentline);
          if (list.isEmpty() || list.last() != loc)
            list.append(loc);
        }
    }
  return list;
}


static int
lua_error_handler(lua_State *L)
{
  QtLuaEngine::Private *d = luaQ_private_noerr(L);
  const char *m = lua_tostring(L, -1);
  if (m && strstr(m, "\nstack traceback:\n\t"))
    return 1;  // hack alert (see lua/src/ldblib.c) 
  luaQ_tracebackskip(L, 1);
  QByteArray message = lua_tostring(L, -1);
  QList<QByteArray> location = find_error_location(L, message);
  bool stateChanged = false;
  if (d)
    {
      d->lastErrorLocation = location;
      d->lastErrorMessage = message;
      d->emitErrorMessage(d->lastErrorMessage);
      QMutexLocker locker(&d->mutex);
      d->errorHandlerFlag = true;
      if (d->pauseOnError && !d->unwindStack)
        stateChanged = lua_pause_sub(L, d, locker);
      d->errorHandlerFlag = false;
    }
  if (stateChanged)
    d->emitStateChanged(QtLuaEngine::Running);
  return 1;
}



bool 
QtLuaEngine::Private::stopHelper(bool unwind)
{
  unwindStack |= unwind;
  lua_Hook hf = lua_gethook(L);
  if (hf != stopHook)
    {
      hookFunction = lua_gethook(L);
      hookMask = lua_gethookmask(L);
      hookCount = lua_gethookcount(L);
    }
  lua_sethook(L, stopHook, LUA_MASKCOUNT|LUA_MASKRET, 1);
  emitStopSignal(); // just to make sure luaQ_doevents() is stopped!
  return true;
}

bool 
QtLuaEngine::Private::resumeHelper(int retcode)
{
  if (pauseLoop)
    pauseLoop->exit(retcode);
  return true;
}


/*! Stops the lua execution as soon as practicable.
  If flag \a nopause if false, the interpreter
  will transition to the paused state until someone
  calls \a resume(). Otherwise  the interpreter stops 
  executing, unwinds the stack, and returns to ready state. */

bool
QtLuaEngine::stop(bool nopause)
{
  QMutexLocker locker(&d->mutex);
  if (d->pauseLoop && nopause)
    return d->resumeHelper(1);
  if (d->rflag && !d->pauseLoop)
    return d->stopHelper(nopause);
  return false;
}


/*! Resume execution after a \a pause().
  Returns \a false when called when the engine is not in paused state. 
  When argument \a nocontinue is true, the interpreter
  stops executing, unwinds the stack, and returns to ready state. */

bool 
QtLuaEngine::resume(bool nocontinue)
{
  QMutexLocker locker(&d->mutex);
  if (d->pauseLoop)
    return d->resumeHelper(nocontinue ? 1 : 0);
  if (d->rflag && !d->pauseLoop && nocontinue)
    return d->stopHelper(true);
  return false;
}


static int
lua_eval_func(lua_State *L)
{
  QtLuaEngine::Private *d = luaQ_private_noerr(L);
  d->errorHandlerFlag = false;
  lua_pushcfunction(L, lua_error_handler);
  int error = luaL_loadstring(L, lua_tostring(L, 1));
  if (! error)
    error = lua_pcall(L, 0, LUA_MULTRET, -2);
  if (!d || d->printResults || (error && d->printErrors))
    luaQ_print(L, lua_gettop(L) - 2);
  if (error)
    lua_error(L);
  return lua_gettop(L) - 2;
}


/*! Evaluate the expression in string \a s.
  This function returns \a true if the evaluation
  was performed without error. It immediately
  returns \a false if the \a QtLuaEngine instance
  state is not "ready" or when one requests
  an asynchronous call from the engine thread.
  Evaluation takes place in the thread 
  owning the \a QtLuaEngine instance.
  Synchronous calls are performed by setting
  flag \a async is \a false. The function waits 
  until the evaluation terminates regardless of 
  the thread from which it is called.
  Asynchronous calls are performed by setting
  flag \a async to \a true and calling this
  function from a thread other than the thread
  owning the \a QtLuaEngine instance. */

bool 
QtLuaEngine::eval(QByteArray s, bool async)
{
  QtLuaLocker lua(this);
  if (! lua.isReady())
    return false;
  if (async && QThread::currentThread() == thread())
    return false;
  lua_settop(L, 0);
  lua_pushcfunction(L, lua_eval_func);
  lua_pushstring(L, s.constData());
  int status = luaQ_pcall(L, 1, 0, 0, this, async);
  return (status == 0);
}


/*! \overload */

bool
QtLuaEngine::eval(QString s, bool async)
{
  return eval(s.toLocal8Bit(), async);
}


/*! Synchronously evaluate string \a s and returns the result.
  This function returns an empty \a QVariantList if
  called when the engine is not in ready state.
  If an error occurs during evaluation, it returns a list
  whose first element is \a QVariant(false) and whose
  second element is the error message. 
  Otherwise is returns a list whose first element is
  \a QVariant(true) and whose remaining elements
  are the evaluation results.
*/

QVariantList 
QtLuaEngine::evaluate(QByteArray s)
{
  QVariantList results;
  QtLuaLocker lua(this);
  if (! lua.isReady())
    return results;
  lua_settop(L, 0);
  lua_pushcfunction(L, lua_eval_func);
  lua_pushstring(L, s.constData());
  int status = luaQ_pcall(L, 1, LUA_MULTRET, 0, this);
  if (status)
    results << QVariant(false);
  else
    results << QVariant(true);
  for (int i = 1; i <= lua_gettop(L); i++)
    results << luaQ_toqvariant(L, i);
  return results;
}


/*! \overload */

QVariantList 
QtLuaEngine::evaluate(QString s)
{
  return evaluate(s.toLocal8Bit());
}




// ========================================
// Calls


struct QtLuaEngine::Catcher : public QObject
{
  Q_OBJECT
public:
  QtLuaEngine::Private *d;
  Catcher(QtLuaEngine::Private *d) : d(d) { }
  virtual bool event(QEvent *e);
public slots:
  void destroy() { delete this; }
};
  

bool
QtLuaEngine::Catcher::event(QEvent *e)
{
  if (e->type() != QEvent::User)
    return false;
  QMutexLocker locker(&d->mutex);
  Q_ASSERT(e == d->hopEvent);
  while (d->lockCount > 0)
    d->hopCondition.wait(&d->mutex);
  // lock
  d->lockThread = QThread::currentThread();
  d->lockCount = 1;
  d->hopEvent = 0;
  bool rflag = d->rflag;
  int stacktop = lua_gettop(d->L);
  int status = LUA_ERRRUN;
  if (d->unwindStack)  
    {
      lua_pushliteral(d->L, "stop (unwinding stack)");
    }
  else try 
    { 
      d->rflag = true;
      locker.unlock();
      if (! rflag)
        d->emitStateChanged(QtLuaEngine::Running);
      status = lua_pcall(d->L, d->hopNA, d->hopNR, d->hopEH);
      locker.relock();
    } 
  catch(...) 
    {
      lua_settop(d->L, stacktop);
      lua_pushliteral(d->L, "uncaught c++ exception");
    }
  d->lockCount = 0;
  if (d->hopLoop) 
    d->hopLoop->exit(status);
  else
    d->emitReadySignal();
  // We use a timer because calling 'delete this' 
  // in an event handler is not supported and 
  // because 'deleteLater' waits until the
  // current eventloop returns...
  QTimer::singleShot(0, this, SLOT(destroy()));
  return true;
}


struct QtLuaEngine::Unlocker : public QObject
{
  Q_OBJECT
public:
  QtLuaEngine::Private *d;
  Unlocker(QtLuaEngine::Private *d) : d(d) {}
  virtual bool event(QEvent *e);
};


bool 
QtLuaEngine::Unlocker::event(QEvent *e) 
{ 
  if (e->type() != QEvent::User)
    return false;
  d->mutex.lock();
  d->lockCount = 0;
  d->hopCondition.wakeOne();
  d->mutex.unlock();
  return true;
}


static int
luaQ_pcall(lua_State *L, int na, int nr, int eh, QObject *obj, bool async)
{
  int status;
  // obtain lua engine
  QtLuaEngine::Private *d = luaQ_private(L);
  d->mutex.lock();
  if (! obj)
    obj = QCoreApplication::instance();
  if (obj->thread() == QThread::currentThread())
    {
      bool rflag = d->rflag;
      d->rflag = true;
      d->mutex.unlock();
      if (! rflag)
        d->emitStateChanged(QtLuaEngine::Running);
      status = lua_pcall(L, na, nr, eh);
    }
  else
    {
      QEvent *event = new QEvent(QEvent::User);
      d->hopEvent = event;
      d->hopNA = na;
      d->hopNR = nr;
      d->hopEH = eh;
      QThread *thread = obj->thread();
      QtLuaEngine::Catcher *catcher = new QtLuaEngine::Catcher(d);
      catcher->moveToThread(thread);
      QCoreApplication::postEvent(catcher, event);
      if (async)
        {
          // This should only be used for async eval.
          // Freeing the last lua locker will wake
          // the catcher waiting on hopCondition
          Q_ASSERT(! d->rflag);
          d->mutex.unlock();
          return 0;
        }
      else
        {
          QEventLoop *savedHopLoop = d->hopLoop;
          int savedLockCount = d->lockCount;
          QThread *savedLockThread = d->lockThread;
          QEventLoop loop;
          d->hopLoop = &loop;
          // The catcher will soon get the hopEvent message
          // and run lua_pcall() in the receiving thread.
          // Then it calls exit() causing us to leave the 
          // event loop and resume execution in this thread.
          // For this to happen we need to unlock the engine
          // and signal hopCondition. But we cannot do it right now
          // because the other thread might complete even before 
          // we call loop.exec(). Posting a message to the Unlocker
          // object ensures that unlock happens after exec() has started.
          // This is not very high-performance :-(.
          QtLuaEngine::Unlocker unlocker(d);
          QEvent *unlockEvent = new QEvent(QEvent::User);
          QCoreApplication::postEvent(&unlocker, unlockEvent);
          d->mutex.unlock();
          status = loop.exec();
          d->mutex.lock();
          d->lockCount = savedLockCount;
          d->lockThread = savedLockThread;
          d->hopLoop = savedHopLoop;
          d->mutex.unlock();
        }
    }
  return status;
}


/*! Thread hopping version of \a lua_pcall().
  This function is similar to \a lua_pcall() but
  arranges the execution to happen in the
  thread of the Qt object \a obj. 
  This only works if the target thread is running an event loop. 
  The current thread then runs an event loop until 
  being notified of the call results.  
  When \a obj is null, the application object
  is assumed (ensuring the code runs in the gui thread). */

int
luaQ_pcall(lua_State *L, int na, int nr, int eh, QObject *obj)
{
  return luaQ_pcall(L, na, nr, eh, obj, false);
}


/*! Convenience function.
    Same as "luaQ_pcall(L,na,nr,0,obj) || lua_error(L)". 
    Unlike \a lua_call() this function calls
    the function with the default error handler
    instead of the current error handler. */

void
luaQ_call(lua_State *L, int na, int nr, QObject *obj)
{
  int base = lua_gettop(L) - na;
  Q_ASSERT(base > 0);
  lua_pushcfunction(L, lua_error_handler);
  lua_insert(L, base);
  int status = luaQ_pcall(L, na, nr, base, obj);
  lua_remove(L, base);
  if (status)
    lua_error(L);
}




static int
call_in_obj_thread(lua_State *L)
{
  int narg = lua_gettop(L);
  // object
  lua_pushvalue(L, lua_upvalueindex(2));
  QObject *obj = luaQ_toqobject(L, -1);
  lua_pop(L, 1); 
  // function
  lua_pushvalue(L, lua_upvalueindex(1));
  Q_ASSERT(lua_isfunction(L, -1));
  lua_insert(L, 1); // function
  // call
  luaQ_call(L, narg, LUA_MULTRET, obj);
  return lua_gettop(L);
}


static int
call_in_arg_thread(lua_State *L)
{
  int narg = lua_gettop(L);
  // object
  QObject *obj = luaQ_toqobject(L, 1);
  if (! obj) 
    luaL_typerror(L, 1, "qobject");
  // function
  lua_pushvalue(L, lua_upvalueindex(1));
  Q_ASSERT(lua_isfunction(L, -1));
  lua_insert(L, 1);
  // call
  luaQ_call(L, narg, LUA_MULTRET, obj);
  return lua_gettop(L);
}



/*! Register functions to be called in the thread of object \a obj.
  If argument \a obj is null, the functions will be called
  in the thread of their first argument (which must be a qobject).
  This is handy to define methods in metaclasses. */

void 
luaQ_register(lua_State *L, const luaL_Reg *l, QObject *obj)
{
  while (l->name)
    {
      lua_pushcfunction(L, l->func);
      if (obj) {
        luaQ_pushqt(L, obj);
        lua_pushcclosure(L, call_in_obj_thread, 2);
      } else
        lua_pushcclosure(L, call_in_arg_thread, 1);
      lua_setfield(L, -2, l->name);
      l += 1;
    }
}





// ========================================
// Bindings


/*! Returns the engine for the current interpreter. */

QtLuaEngine *
luaQ_engine(lua_State *L)
{
  QtLuaEngine::Private *d = luaQ_private(L);
  return d->q;
}


/*! Pushes the qt package table on the stack. */

void
luaQ_pushqt(lua_State *L)
{
  lua_pushlightuserdata(L, (void*)qtKey);
  lua_rawget(L, LUA_REGISTRYINDEX);
  Q_ASSERT(lua_istable(L, -1));
}


/* conversions and tests */

static QVariant *
luaQ_toqvariantp(lua_State *L, int index)
{
  QVariant *vp = 0;
  if (lua_isuserdata(L, index) && 
      lua_getmetatable(L, index))
    {
      lua_rawget(L, LUA_REGISTRYINDEX);
      void *v = lua_touserdata(L, -1);
      lua_pop(L, 1);
      if (v == (void*)qtKey)
        vp = static_cast<QVariant*>(lua_touserdata(L, index));
    }
  return vp;
}


static inline bool
qvariant_has_object_type(const QVariant *vp)
{
  int type = vp->userType();
  return (type == qMetaTypeId<QObjectPointer>() ||
          type == QMetaType::QObjectStar ||
          type == QMetaType::QWidgetStar );
}


static inline QObject *
qvariant_to_object(const QVariant *vp)
{
  int type = vp->userType();
  if (type == qMetaTypeId<QObjectPointer>())
    return *static_cast<QObjectPointer const *>(vp->constData());
  if (type == QMetaType::QObjectStar)
    return *static_cast<QObject * const *>(vp->constData());
  return 0;
}


static inline QObject *
qvariant_to_object(const QVariant *vp, const QMetaObject *mo)
{
  QObject *obj = qvariant_to_object(vp);
  if (obj && mo)
    {
      const QMetaObject *m = obj->metaObject();
      while (m && m != mo)
        m = m->superClass();
      if (m != mo)
        obj = 0;
    }
  return obj;
}


/*! Extract \a QVariant from the lua value located 
  at position \a index in the stack. 
  Standard lua types are converted to \a QVariant
  as needed. */

QVariant 
luaQ_toqvariant(lua_State *L, int index, int type)
{
  QVariant v;
  switch (lua_type(L, index))
    {
    case LUA_TBOOLEAN:
      v = QVariant((bool)lua_toboolean(L, index));
      break;
    case LUA_TNUMBER:
      v = QVariant((double)lua_tonumber(L, index));
      break;
    case LUA_TUSERDATA: {
      const QVariant *vp = 0;
      if (! (vp = luaQ_toqvariantp(L, index)))
        break;
      if (qvariant_has_object_type(vp) && !qvariant_to_object(vp))
        break;
      v = *vp;
      break; }
    case LUA_TSTRING: {
      size_t l; 
      const char *s = lua_tolstring(L, index, &l);
      v = QVariant(QByteArray(s, l)); 
      break; }
    default:
      break;
    }
  if  (type)
    {
      int vtype = v.userType();
      if (type == vtype)
        return v;
      else if (type == QVariant::ByteArray && vtype == QVariant::String)
        v = v.toString().toLocal8Bit();
      else if (type == QVariant::String && vtype == QVariant::ByteArray)
        v = QString::fromLocal8Bit(v.toByteArray().constData());
      else if (! v.convert(QVariant::Type(type)))
        v = QVariant();
    }
  return v;
}


/*! Returns true if the lua value at position \a index
  points to a \a QObject that inherits from the class
  identified by metaobject \a mo. */

QObject*
luaQ_toqobject(lua_State *L, int index, const QMetaObject *mo)
{
  const QVariant *vp = luaQ_toqvariantp(L, index);
  return (vp) ? qvariant_to_object(vp, mo) : 0;
}



/* This part of the protector makes sure that
   certain implicit shared objects are
   destroyed in the gui thread instead
   of the thread that destroys the variant.
   Yes this is tricky. */

bool 
QtLuaEngine::Protector::maybeProtect(const QVariant &var)
{
  if (QThread::currentThread() == QCoreApplication::instance()->thread())
    return false;
  int type = var.userType();
  switch(type)
    {
    case QMetaType::QPixmap:
    case QMetaType::QBrush:
    case QMetaType::QPen:
      return protect(var);
    default:
      return false;
    }
}

bool 
QtLuaEngine::Protector::protect(const QVariant &var)
{
  QMutexLocker lock(&mutex);
  int size = saved.size();
  saved.append(var);
  if (! size)
    QCoreApplication::postEvent(this, new QEvent(QEvent::User));
  return true;
}

bool 
QtLuaEngine::Protector::event(QEvent *e)
{
  if (e->type() == QEvent::User)
    {
      QMutexLocker lock(&mutex);
      saved.clear(); // possible actual deletion 
      return true;
    }
  return QObject::event(e);
}



/* metatable material */

static int
luaQ_m__gc(lua_State *L)
{
  QtLuaEngine::Private *d = luaQ_private_noerr(L);
  QObject *obj = luaQ_toqobject(L, 1);
  if (obj && d && d->isObjectLuaOwned(obj))
    obj->deleteLater();
  QVariant *vp = luaQ_toqvariantp(L, 1);
  if (vp && d && d->protector)
    d->protector->maybeProtect(*vp);
  if (vp)
    vp->QVariant::~QVariant();
  return 0;
}


static int
luaQ_m__type(lua_State *L)
{
  // LUA: "_val_:type()"
  // Returns a string describing the type of a qt value.
  // Returns nil if _val_ is not a qt value.
  if (lua_isuserdata(L, 1) && 
      lua_getmetatable(L, 1))
    {
      lua_pushliteral(L, "__typename");
      lua_rawget(L, -2);
      if (lua_isstring(L, -1))
        return 1;
    }
  lua_pushstring(L, luaL_typename(L, 1));
  return 1;
}


static int
luaQ_m__isa(lua_State *L)
{
  // LUA: "_val_:isa(_str_)"
  // Returns true is _val_ is an instance of class _str_.
  // In the case of objects, the classname may or may not
  // have a final star.
  bool b = false;
  QObject *obj;
  const QVariant *vp = luaQ_toqvariantp(L, 1);
  const char *t = luaL_checkstring(L, 2);
  if (! vp)
    b = (!strcmp(t, lua_typename(L, lua_type(L, 1))));
  else if (! qvariant_has_object_type(vp))
    b = (!strcmp(t, QMetaType::typeName(vp->userType())) ||
         !strcmp(t, vp->typeName()) );
  else if ((obj = luaQ_toqobject(L, 1)))
    {
      const QMetaObject *mo = obj->metaObject();
      int tlen = strlen(t);
      if (tlen>0 && t[tlen-1]=='*')    
        tlen -= 1;
      while (mo && !b)
        {
          const char *c = mo->className();
          b = (!strncmp(t, c, tlen) && c[tlen]==0);
          mo = mo->superClass();
        }
    }
  lua_pushboolean(L, b);
  return 1;
}


static int
luaQ_m__eq(lua_State *L)
{
  // __eq in metatable
  const QVariant *vp1 = luaQ_toqvariantp(L, 1);
  const QVariant *vp2 = luaQ_toqvariantp(L, 2);
  bool ok = false;
  if (*vp1 == *vp2)
    ok = true;
  else if (qvariant_has_object_type(vp1) &&
           qvariant_has_object_type(vp2) &&
           qvariant_to_object(vp1) == qvariant_to_object(vp2) )
    ok = true;
  lua_pushboolean(L, ok);
  return 1;
}


static int
luaQ_m__tostring(lua_State *L)
{
  // LUA: "_val_:tostring()"
  // Converts a qt value to a nice printable string.
  // Returns nil if _val_ is not a qt value.
  const QVariant *vp = luaQ_toqvariantp(L, 1);
  if (! vp)
    lua_pushnil(L);
  else if (qvariant_has_object_type(vp))
    {
      QObject *obj = qvariant_to_object(vp);
      if (obj)
        lua_pushfstring(L, "qt.%s (%p)", obj->metaObject()->className(), obj);
      else
        lua_pushliteral(L, "qt.zombie");
    }
  else
    {
      int type = vp->userType();
      const void *ptr = vp->constData();
      QVariant var = *vp;
      if (var.type() == QVariant::String)
        lua_pushstring(L, var.toString().toLocal8Bit().constData());
      else if (var.convert(QVariant::ByteArray))
        lua_pushstring(L, var.toByteArray().constData());
      else
        lua_pushfstring(L, "qt.%s (%p)", QMetaType::typeName(type), ptr);
    }
  return 1;
}


static int
luaQ_m__tonumber(lua_State *L)
{
  // LUA: "_val_:tonumber()"
  // Converts a qt value to a number.
  // Returns nil if the conversion is impossible
  QVariant var = luaQ_toqvariant(L, 1);
  if (var.convert(QVariant::Double))
    lua_pushnumber(L, var.toDouble());
  else
    lua_pushnil(L);
  return 1;
}


static int
luaQ_m__tobool(lua_State *L)
{
  // LUA: "_val_:tobool()"
  // Converts a qt value to a boolean.
  const QVariant *vp = luaQ_toqvariantp(L, 1);
  if (! vp)
    lua_pushnil(L);
  else if (qvariant_has_object_type(vp))
    lua_pushboolean(L, !!qvariant_to_object(vp));
  else
    lua_pushboolean(L, !vp->isNull());
  return 1;
}


static int
luaQ_m__call(lua_State *L)
{
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_pushliteral(L, "new");
  lua_rawget(L, 1);
  if (!lua_isfunction(L, -1))
    luaL_error(L, "constructor 'new' not found");
  lua_replace(L, 1);
  lua_call(L, lua_gettop(L)-1, LUA_MULTRET);
  return lua_gettop(L);
}


/* metatable material: get/set property */

static int
luaQ_m__getsetproperty(lua_State *L)
{
  const QVariant* vp = luaQ_toqvariantp(L, lua_upvalueindex(1));
  if (vp->userType() != qMetaTypeId<QtLuaPropertyInfo>())
    luaL_error(L, "internal error while accessing property");
  const QtLuaPropertyInfo *info =
    static_cast<const QtLuaPropertyInfo*>(vp->constData());
  const QMetaProperty &mp = info->metaProperty;
  QObject *obj = luaQ_toqobject(L, 1, info->metaObject);
  if (! obj)
    luaL_typerror(L, 0, info->metaObject->className());
  if (! mp.isScriptable(obj))
    luaL_error(L, "property " LUA_QS " is not scriptable", mp.name());
  if (lua_gettop(L) == 1)
    {
      if (! mp.isReadable())
        luaL_error(L, "property " LUA_QS " is not readable", mp.name());
      // get property
      QVariant v = mp.read(obj);
      // convert enum codes into enum names
      QByteArray n;
      if (mp.isFlagType() && v.canConvert(QVariant::Int))
        v = mp.enumerator().valueToKeys(v.toInt());
      else if (mp.isEnumType() && v.canConvert(QVariant::Int))
        v = mp.enumerator().valueToKey(v.toInt());
      // return
      luaQ_pushqt(L, v);
      return 1;
    }
  else
    {
      QVariant v = luaQ_toqvariant(L, 2);
      if (! mp.isWritable())
        luaL_error(L, "property " LUA_QS " is not writable", mp.name());
      // string conversion
      if (mp.type() == QVariant::String && v.type() == QVariant::ByteArray)
        v = QString::fromLocal8Bit(v.toByteArray());
      // enum and flags conversion
      if (mp.isFlagType() && v.type() == QVariant::ByteArray)
        {
          QMetaEnum me = mp.enumerator();
          int n = me.keysToValue(v.toByteArray().constData());
          if (n == -1)
            luaL_error(L, "unrecognized flag for " LUA_QS, me.name());
          v = QVariant(n);
        }
      else if (mp.isEnumType() && v.type() == QVariant::ByteArray)
        {
          QMetaEnum me = mp.enumerator();
          int n = me.keyToValue(v.toByteArray().constData());
          if (n == -1)
            luaL_error(L, "unrecognized enum for " LUA_QS, me.name());
          v = QVariant(n);
        }
      // check types
      if (! mp.write(obj, v))
        luaL_error(L, "cannot convert value for property '%s'", mp.name());
    }
  return 0;
}


/* metatable material: invoking methods */

static void *
make_argtype(QByteArray type)
{
  int tid = QMetaType::type(type);
  if (type.endsWith("*"))
    {
      type.chop(1);
      const QMetaObject *mo = qtLuaEngineGlobal()->findMetaObject(type);
      if (mo)
        return (void*)mo;
      else if (!tid)
        tid = QMetaType::VoidStar;
    }
  if (tid)
    return (void*)((((size_t)tid)<<1)|1);
  return 0;
}


static inline int
v_to_type(void *v)
{
  return (((size_t)v)&1) ? (((size_t)v)>>1) : 0;
}


static inline const QMetaObject *
v_to_metaobject(void *v)
{
  return (((size_t)v)&1) ? 0 : static_cast<const QMetaObject*>(v);
}


static const char *
v_to_name(void *v)
{
  int type = v_to_type(v);
  if (type)
    return QMetaType::typeName(type);
  else if (v)
    return v_to_metaobject(v)->className();
  else
    return "void";
}


static void *
construct_arg(QVariant &var, void *vtype)
{
  int type = v_to_type(vtype);
  const QMetaObject *mo = v_to_metaobject(vtype);
  QObject *obj;
  if (type == QVariant::String && var.type() == QVariant::ByteArray)
    var = QString::fromLocal8Bit(var.toByteArray());
  if (type == qMetaTypeId<QVariant>())
    return static_cast<void*>(new QVariant(var));
  else if (type && (var.userType() == type || var.convert(QVariant::Type(type))))
    return QMetaType::construct(type, var.constData());
  else if (mo && (obj = qvariant_to_object(&var)))
    return static_cast<void*>(new QObject*(obj));
  return 0;
}


static void
destroy_arg(void *arg,  void *vtype)
{
  int type = v_to_type(vtype);
  const QMetaObject *mo = v_to_metaobject(vtype);
  if (! arg)
    return;
  else if (type == qMetaTypeId<QVariant>())
    delete static_cast<QVariant*>(arg);
  else if (type)
    QMetaType::destroy(type, arg);
  else if (mo)
    delete static_cast<QObject**>(arg);
}


static void *
construct_retval(void *vtype)
{
  int type = v_to_type(vtype);
  const QMetaObject *mo = v_to_metaobject(vtype);
  if (mo)
    return static_cast<void*>(new QObject*(0));
  else if (type == QMetaType::Void)
    return 0;
  else if (type == qMetaTypeId<QVariant>())
    return static_cast<void*>(new QVariant());
  else if (type)
    return QMetaType::construct(type);
  return 0;
}


static int
luaQ_p_push_retval(lua_State *L, void *arg, void *vtype)
{
  int type = v_to_type(vtype);
  const QMetaObject *mo = v_to_metaobject(vtype);
  if (! arg)
    return 0;
  if (type == qMetaTypeId<QVariant>())
    luaQ_pushqt(L, *static_cast<QVariant*>(arg));
  else if (type)
    luaQ_pushqt(L, QVariant(type, arg));
  else if (mo)
    luaQ_pushqt(L, *static_cast<QObject**>(arg));
  else
    return 0;
  return 1;
}


static inline int
type_class(int type)
{
  switch(type)
    {
    case QMetaType::Double:
    case QMetaType::Float:
    case QMetaType::Char:
    case QMetaType::Short:
    case QMetaType::Int:
    case QMetaType::Long:
    case QMetaType::LongLong:
    case QMetaType::UChar:
    case QMetaType::UShort:
    case QMetaType::UInt:
    case QMetaType::ULong:
    case QMetaType::ULongLong:
      return -1; // numerical
    case QMetaType::QByteArray:
    case QMetaType::QString:
      return -2; // string
    default:
      return type;
    }
}


static int
select_overload(VarVector &vars, const QtLuaMethodInfo *info)
{
  // fast path when single method
  int overloads = info->d.size();
  if (overloads == 1)
    return 0;
  // fast path when single method with right number of args
  int i;
  int b = -1;
  int bn = 0;
  int narg = vars.size(); 
  for (i=0; i<overloads; i++)
    if (narg == info->d[i].types.size())
      { b = i; bn += 1; }
  if (b >= 0 && bn == 1)
    return b;
  // match types
  b = -1;
  bn = 0;
  int bs = INT_MAX;
  for (i=0; i<overloads; i++)
    if (narg == info->d[i].types.size())
      {
        int j;
        int s = 0;
        const QtLuaMethodInfo::Detail &d = info->d[i];
        for (j=1; j<narg; j++)
          {
            int type = v_to_type(d.types[j]);
            int argtype = vars[j].userType();
            const QMetaObject *mo = v_to_metaobject(d.types[j]);
            if (type == qMetaTypeId<QVariant>())
              continue;
            if (type && type == argtype)
              continue;
            if (mo && qvariant_to_object(&vars[j], mo))
              continue;
            s += 1;
            if (type && type_class(type) == type_class(argtype))
              continue;
            s += 10;
            QVariant var = vars[j];
            if (var.canConvert(QVariant::Type(type)))
              continue;
            s += INT_MAX / 2;
            break;
          }
        if (j < narg)
          continue;
        if (s == bs)
          bn += 1;
        else if (s < bs)
          { b = i; bs = s; bn = 1; }
      }
  if (b >= 0 && bn == 1)
    return b;
  return -1;
}


static int
luaQ_m__invokemethod(lua_State *L)
{
  int i;
  const int narg = lua_gettop(L) - 1;
  const QVariant* vp = luaQ_toqvariantp(L, narg  + 1);
  if (vp->userType() != qMetaTypeId<QtLuaMethodInfo>())
    luaL_error(L, "internal error while invoking method");
  const QtLuaMethodInfo *info 
    = static_cast<const QtLuaMethodInfo*>(vp->constData());
  QObject *obj = luaQ_toqobject(L, 1, info->metaObject);
  if (! obj)
    luaL_error(L, "bad 'self' argument (%s expected)", 
               info->metaObject->className());
  VarVector vargs(narg);
  for (i=1; i<narg; i++)
    vargs[i] = luaQ_toqvariant(L, i+1);
  int m = select_overload(vargs, info);
  if (m < 0)
    luaL_error(L, "cannot resolve ambiguous overload");
  const QtLuaMethodInfo::Detail &d = info->d[m];
  PtrVector pargs(narg);
  int arg = 0;
  int errarg = -1;
  int nret = 0;
  try
    {
      pargs[arg++] = construct_retval(d.types[0]);
      for (int i=1; i<narg; i++)
        if (! (pargs[arg++] = construct_arg(vargs[i], d.types[i])))
          { errarg = i; break ; }
      // invoke
      if (errarg < 0)
        {
          obj->qt_metacall(QMetaObject::InvokeMetaMethod, d.id, pargs.data());
          nret = luaQ_p_push_retval(L, pargs[0], d.types[0]);
        }
      // deallocate
      while (--arg >= 0)
        destroy_arg(pargs[arg], d.types[arg]);
      // diagnosis
      if (errarg >= 0)
        luaL_error(L, "bad argument #%d (%s expected)", 
                   errarg, v_to_name(d.types[errarg]) );
    }
  catch(...)
    {
      while (--arg >= 0)
        destroy_arg(pargs[arg], d.types[arg]);
      throw;
    }
  return nret;
}


static int
luaQ_m__call_invokemethod(lua_State *L)
{
  QObject *obj = luaQ_toqobject(L, 1);
  if (! obj)
    luaL_error(L, "bad 'self' argument (not a qobject)");
  int narg = lua_gettop(L);
  lua_pushcfunction(L, luaQ_m__invokemethod);
  lua_insert(L, 1);
  lua_pushvalue(L, lua_upvalueindex(1));
  luaQ_call(L, narg + 1, LUA_MULTRET, obj);
  return lua_gettop(L);
}


/* metatable material: index/newindex */

static int
luaQ_m__index(lua_State *L)
{
  // __index in metatable
  // Check arguments
  QObject *obj = luaQ_checkqobject<QObject>(L, 1);
  if (lua_getmetatable(L, 1)) 
    {
      lua_pushliteral(L, "__metatable");
      lua_rawget(L, -2);
      // ..stack: object key metatable metaclass
      // Search metaclass
      if (lua_istable(L, -1))
        {
          lua_pushvalue(L, 2);
          lua_gettable(L, -2);
          // ..stack: object key metatable metaclass value
          const QVariant *vp = luaQ_toqvariantp(L, -1);
          int type = (vp) ? vp->userType() : QMetaType::Void;
          if (type == qMetaTypeId<QtLuaMethodInfo>())
            {
              lua_pushcclosure(L, luaQ_m__call_invokemethod, 1);
              return 1;
            }
          else if (type == qMetaTypeId<QtLuaPropertyInfo>())
            {
              lua_pushcclosure(L, luaQ_m__getsetproperty, 1);
              lua_pushvalue(L, 1);
              luaQ_call(L, 1, 1, obj);
              return 1;
            }
          else if (! lua_isnil(L, -1))
            return 1;
          lua_pop(L, 1);
        }
      lua_pop(L, 2);
    }
  // ..stack: object key
  // Search children
  const char *key = luaL_checkstring(L, 2);
  QString name = QString::fromLocal8Bit(key);
  foreach (QObject *o, obj->children())
    if (o->objectName() == name)
      {
        luaQ_pushqt(L, o);
        return 1;
      }
  QObject *o = qFindChild<QObject*>(obj, QString::fromLocal8Bit(key));
  if (o)
    {
      luaQ_pushqt(L, o);
      return 1;
    }
  // Failed
  lua_pushnil(L);
  return 1;
}


static int
luaQ_m__newindex(lua_State *L)
{
  // __newindex in metatables
  // ..stack: object key value
  QObject *obj = luaQ_checkqobject<QObject>(L, 1);
  if (lua_getmetatable(L, 1))
    {
      lua_pushliteral(L, "__metatable"); 
      lua_rawget(L, -2);
      // ..stack: object key value metatable metaclass
      // Search property
      if (lua_istable(L, -1))
        {
          lua_pushvalue(L, 2);
          lua_gettable(L, -2);
          // ..stack: object key value metatable metaclass curval
          const QVariant *vp = luaQ_toqvariantp(L, -1);
          int type = (vp) ? vp->userType() : QMetaType::Void;
          if (type == qMetaTypeId<QtLuaPropertyInfo>())
            {
              lua_pushcclosure(L, luaQ_m__getsetproperty, 1);
              lua_pushvalue(L, 1);
              lua_pushvalue(L, 3);
              luaQ_call(L, 2, 0, obj);
              return 0;
            }
          lua_pop(L, 1);
        }
      lua_pop(L, 2);
    }
  // Failed
  const char *key = luaL_checkstring(L, 2);
  luaL_error(L, "cannot set unrecognized property '%s'", key);
  return 0;
}


/* metaclasses */

static const luaL_Reg qtval_lib[] = {
  { "tostring", luaQ_m__tostring },
  { "tonumber", luaQ_m__tonumber },
  { "tobool", luaQ_m__tobool },
  { "type", luaQ_m__type },
  { "isa", luaQ_m__isa },
  {NULL, NULL}
};


void
luaQ_buildmetaclass(lua_State *L, int type)
{
  // Create table
  lua_createtable(L, 0, 0); 
  // Fill table
  lua_pushvalue(L, -1);
  lua_setfield(L, -2, "__index");
  lua_pushcfunction(L, luaQ_m__call);
  lua_setfield(L, -2, "__call");
  if (type != QMetaType::Void)
    {
      luaQ_pushmeta(L,  QMetaType::Void);
      lua_pushliteral(L, "__metatable");
      lua_rawget(L, -2);
      Q_ASSERT(lua_istable(L, -1));
      // stack: class metasuper classsuper
      lua_setmetatable(L, -3);
      lua_pop(L, 1);
    }
  else
    {
      // Standard methods
      luaL_register(L, 0, qtval_lib);
    }
  // Insert class into qt package
  lua_pushlightuserdata(L, (void*)qtKey);
  lua_rawget(L, LUA_REGISTRYINDEX); 
  Q_ASSERT(lua_istable(L, -1));
  lua_pushstring(L, QMetaType::typeName(type));
  lua_pushvalue(L, -3);
  // ..stack: metaclass qtkeytable typename metaclass
  lua_rawset(L, -3);
  lua_pop(L, 1);
}


static PtrVector 
make_argtypes(QMetaMethod method)
{
  PtrVector types;
  QList<QByteArray> typeNames = method.parameterTypes();
  types += make_argtype(method.typeName());
  foreach(QByteArray b, typeNames)
    types += make_argtype(b);
  types.squeeze();
  return types;
}


void
luaQ_buildmetaclass(lua_State *L, const QMetaObject *mo)
{
  // Create table
  lua_createtable(L, 0, 0);
  // Fill table
  lua_pushvalue(L, -1);
  lua_setfield(L, -2, "__index");
  lua_pushcfunction(L, luaQ_m__call);
  lua_setfield(L, -2, "__call");
  const QMetaObject *super = mo->superClass();
  if (super)
    {
      // Plug superclass into index
      luaQ_pushmeta(L, super);
      lua_pushliteral(L, "__metatable");
      lua_rawget(L, -2);
      Q_ASSERT(lua_istable(L, -1));
      // stack: class metasuper classsuper
      lua_setmetatable(L, -3);
      lua_pop(L, 1);
    }
  else
    {
      // Standard methods
      luaL_register(L, 0, qtval_lib);
    }
  // Slots and invokable method
  QMap<QByteArray,QtLuaMethodInfo> overloads;
  int fm = mo->methodOffset();
  int lm = mo->methodCount();
  for  (int i=fm; i<lm; i++)
    {
      QMetaMethod method = mo->method(i);
      QByteArray sig = method.signature();
      if (method.access() != QMetaMethod::Private)
        {
          QtLuaMethodInfo::Detail d;
          d.id = i;
          d.types = make_argtypes(method);
          QtLuaMethodInfo info;
          info.metaObject = mo;
          info.d += d;
          info.d.squeeze();
          lua_pushstring(L, sig.constData());
          luaQ_pushqt(L, qVariantFromValue(info));
          // stack: class signature methodinfo
          lua_rawset(L, -3);
          // record overloads
          int len = sig.indexOf('(');
          if (len > 0)
            {
              sig = sig.left(len);
              if (! overloads.contains(sig))
                overloads[sig] = info;
              else {
                QtLuaMethodInfo &oinfo = overloads[sig];
                oinfo.d += d;
              }
            }
        }
    }
  // Overloaded methods
  QMap<QByteArray,QtLuaMethodInfo>::const_iterator it;
  for (it = overloads.constBegin(); it != overloads.constEnd(); ++it)
    {
      QtLuaMethodInfo info = it.value();
      info.d.squeeze();
      lua_pushstring(L, it.key().constData());
      luaQ_pushqt(L, qVariantFromValue(info));
      lua_rawset(L, -3);
    }
  // Properties
  int fp = mo->propertyOffset();
  int lp = mo->propertyCount();
  for  (int j=fp; j<lp; j++)
    {
      QtLuaPropertyInfo info;
      info.id = j;
      info.metaObject = mo;
      info.metaProperty = mo->property(j);
      lua_pushstring(L, info.metaProperty.name());
      luaQ_pushqt(L, qVariantFromValue(info));
      // stack: class name propinfo
      lua_rawset(L, -3);
    }
  // Insert class into qt package
  lua_pushlightuserdata(L, (void*)qtKey);
  lua_rawget(L, LUA_REGISTRYINDEX); 
  Q_ASSERT(lua_istable(L, -1));
  lua_pushstring(L, mo->className());
  lua_pushvalue(L, -3);
  // ..stack: metaclass qtkeytable classname metaclass
  lua_rawset(L, -3);
  lua_pop(L, 1);
}


static const luaL_Reg qtmeta_lib[] = {
  { "__tostring", luaQ_m__tostring },
  { "__eq", luaQ_m__eq },
  { "__gc", luaQ_m__gc },
  {NULL, NULL}
};


void
luaQ_fillmetatable(lua_State *L, int type, const QMetaObject *mo)
{
  // Fill common entries
  luaL_register(L, 0, qtmeta_lib);
  // Fill distinct entries
  if (mo)
    {
      luaQ_buildmetaclass(L, mo);
      lua_setfield(L, -2, "__metatable");
      lua_pushcfunction(L, luaQ_m__index);
      lua_setfield(L, -2, "__index");
      lua_pushcfunction(L, luaQ_m__newindex);
      lua_setfield(L, -2, "__newindex");
      lua_pushfstring(L, "%s*", mo->className());
      lua_setfield(L, -2, "__typename");
    }
  else
    {
      luaQ_buildmetaclass(L, type);
      lua_pushvalue(L, -1);
      lua_setfield(L, -3, "__index");
      lua_setfield(L, -2, "__metatable");
      lua_pushstring(L, QMetaType::typeName(type));
      lua_setfield(L, -2, "__typename");
    }
}


/*! Pushes the metatable for qt values with type id \a type. */

void 
luaQ_pushmeta(lua_State *L, int type)
{
  lua_pushlightuserdata(L, (void*)metaKey);
  lua_rawget(L, LUA_REGISTRYINDEX);
  Q_ASSERT(lua_istable(L, -1));
  lua_pushlightuserdata(L, (void*)((((size_t)type)<<1)|1));
  lua_rawget(L, -2);
  // stack: metakeytable metatable
  if (lua_istable(L,-1))
    {
      lua_remove(L, -2);
    }
  else
    {
      lua_createtable(L, 0, 6);
      // stack: metakeytable nil metatable
      lua_pushlightuserdata(L, (void*)((((size_t)type)<<1)|1));
      lua_pushvalue(L, -2);
      lua_rawset(L, -5);
      lua_pushvalue(L, -1);
      lua_pushlightuserdata(L, (void*)qtKey);
      lua_rawset(L, LUA_REGISTRYINDEX);
      // fill metatable
      luaQ_fillmetatable(L, type, 0);
      lua_replace(L, -3);
      lua_pop(L, 1);
    }
}


/*! Pushes the metatable for qt objects with meta object \a mo. */

void 
luaQ_pushmeta(lua_State *L, const QMetaObject *mo)
{
  lua_pushlightuserdata(L, (void*)metaKey);
  lua_rawget(L, LUA_REGISTRYINDEX);
  Q_ASSERT(lua_istable(L, -1));
  lua_pushlightuserdata(L, (void*)mo);
  lua_rawget(L, -2);
  // ..stack: metakeytable metatable
  if (lua_istable(L,-1))
    {
      lua_remove(L, -2);
    }
  else
    {
      // record metaobject name
      qtLuaEngineGlobal()->registerMetaObject(mo, false);
      // create metatable
      lua_createtable(L, 0, 6);
     // stack: metakeytable maybehook metatable
      lua_pushlightuserdata(L, (void*)mo);
      lua_pushvalue(L, -2);
      lua_rawset(L, -5);
      lua_pushvalue(L, -1);
      lua_pushlightuserdata(L, (void*)qtKey);
      lua_rawset(L, LUA_REGISTRYINDEX);
      // fill metatable
      luaQ_fillmetatable(L, 0, mo);
      lua_replace(L, -3);
      lua_pop(L, 1);
    }
}


/*! Pushes the metatable for qobject \a o. */

void 
luaQ_pushmeta(lua_State *L, const QObject *o)
{
  luaQ_pushmeta(L, o->metaObject());
}


/*! Pushes value \a var onto the stack.
  Common variant types are converted 
  to standard lua types when applicable. */

void 
luaQ_pushqt(lua_State *L, const QVariant &var)
{
  switch(var.userType())
    {
    case QVariant::Invalid:
      lua_pushnil(L);
      break;
    case QVariant::Bool:
      lua_pushboolean(L, var.toBool());
      break;
    case QVariant::Double:
    case QVariant::Int:
    case QVariant::ULongLong:
    case QVariant::UInt:
    case QMetaType::Float:
    case QMetaType::Char:
    case QMetaType::Short:
    case QMetaType::Long:
    case QMetaType::UChar:
    case QMetaType::UShort:
    case QMetaType::ULong:
      lua_pushnumber(L, var.toDouble());
      break;
    case QVariant::ByteArray:
      {
        QByteArray b = var.toByteArray();
        lua_pushlstring(L, b.constData(), b.size());
        break; 
      }
    default: 
      if (qvariant_has_object_type(&var))
        {
          luaQ_pushqt(L, qvariant_to_object(&var));
        }
      else
        {
          void *v = lua_newuserdata(L, sizeof(QVariant));
          new (v) QVariant(var);
          luaQ_pushmeta(L, var.userType());
          lua_setmetatable(L, -2);
          break; 
        }
    }
}


/*! Pushes qobject \obj onto the stack.
  Flag \a owned indicates whether the object should
  be deleted when lua deallocates the object descriptor.
  Null object pointers are converted to lua \a nil value. */

void 
luaQ_pushqt(lua_State *L, QObject *obj, bool owned)
{
  if (! obj)
    {
      lua_pushnil(L);
    }
  else
    {
      // Search already created objects
      lua_pushlightuserdata(L, (void*)objectKey);
      lua_rawget(L, LUA_REGISTRYINDEX);
      Q_ASSERT(lua_istable(L, -1));
      lua_pushlightuserdata(L, (void*)obj);
      lua_rawget(L, -2);
      // ..stack: objecttable object
      if (lua_isuserdata(L, -1))
        {
          lua_remove(L, -2);
        }
      else
        {
          lua_pop(L, 1);
          // ..stack: objecttable
          // Make new userdata for object
          QObjectPointer objp = obj;
          QVariant objv = qVariantFromValue(objp);
          void *v = lua_newuserdata(L, sizeof(QVariant));
          new (v) QVariant(objv);
          // Fill object
          luaQ_pushmeta(L, obj->metaObject());
          lua_setmetatable(L, -2);
          // ..stack: objecttable object
          // Record object
          lua_pushlightuserdata(L, (void*)obj);
          lua_pushvalue(L, -2);
          // ..stack: objecttable object ud object
          lua_rawset(L, -4);
          // ..stack: objecttable object
          lua_remove(L, -2);
          // ..stack: object
          if (owned)
            {
              QtLuaEngine::Private *d = luaQ_private(L);
              if (d) 
                d->makeObjectLuaOwned(obj);
            }
        }
    }
}



// ========================================
// Signal interception


struct QtLuaEngine::Receiver : public QObject
{
  Q_OBJECT
public:
  virtual ~Receiver();
  Receiver(QtLuaEngine *q) : 
    QObject(q), d(q->d), sender(0), direct(false), args(0) {}
public slots:
  void universal();
  void disconnect();
  bool connect(QObject *sender, const char *signal, bool direct);
public:
  QPointer<QtLuaEngine::Private> d;
  QPointer<QObject> sender;
  QByteArray signature;
  PtrVector types;
  bool direct;
protected:
  void **args;
};


struct QtLuaEngine::Receiver2
#ifndef Q_MOC_RUN
  : public QtLuaEngine::Receiver
#endif
{
  Receiver2(QtLuaEngine *engine) : QtLuaEngine::Receiver(engine) {}
  virtual int qt_metacall(QMetaObject::Call, int, void**);
};


int 
QtLuaEngine::Receiver2::qt_metacall(QMetaObject::Call c, int id, void **a)
{
  void **old = args;
  args = a;
  id = Receiver::qt_metacall(c,id,a);
  args = old;
  return id;
}


QtLuaEngine::Receiver::~Receiver()
{
  if (d && sender)
    {
      QtLuaQueuedSignal q;
      q.delsignal = (void*)this;
      QMutexLocker locker(&d->mutex);
      d->queuedSignals += q;
    }
}


void
QtLuaEngine::Receiver::disconnect()
{
  if (sender)
    QObject::disconnect(sender, 0, this, 0);
}


bool
QtLuaEngine::Receiver::connect(QObject *obj, const char *sig, bool d)
{
  Q_ASSERT(sig);
  Q_ASSERT(obj);
  const QMetaObject *m = obj->metaObject();
  Q_ASSERT(m);
  if (sender)
    return false;
  // search signal
  if (sig[0]>= '0' && sig[0]<='3')
    sig = sig + 1;
  signature = sig;
  int i = m->indexOfSignal(signature.constData());
  if (i < 0)
    {
      signature = QMetaObject::normalizedSignature(sig);
      i = m->indexOfSignal(signature.constData());
      if (i < 0)
        return false;
    }
  // metamethod
  QMetaMethod method = m->method(i);
  if (method.methodType() != QMetaMethod::Signal)
    return false;
  // setup
  sender = obj;
  types = make_argtypes(m->method(i));
  signature.prepend('0' + QSIGNAL_CODE);
#if QT_VERSION >= 0x040800
  const int sigIndex = i;
  const int memberOffset = QObject::staticMetaObject.methodCount();
  bool okay = QMetaObject::connect(obj, sigIndex, this, memberOffset,
                                   Qt::DirectConnection, 0);
#else
  bool okay = QObject::connect(obj, signature.constData(), this,
                               SLOT(universal()), Qt::DirectConnection);
#endif
  if (okay)
    {
      QObject::connect(obj, SIGNAL(destroyed(QObject*)), 
                       this, SLOT(deleteLater()) );
      direct = d;
      return true;
    }
  else
    {
      sender = 0;
      signature.clear();
      types.clear();
      return false;
    }
}


void
QtLuaEngine::Receiver::universal()
{
  bool queueSignal = false;
  QThread *mythread = QThread::currentThread();
  QMutexLocker locker(&d->mutex);
  if (d && direct && !d->hopEvent && 
      d->lockCount>0 && d->lockThread == mythread)
    {
      // find closure
      lua_State *L = d->L;
      int base = lua_gettop(L);
      lua_pushlightuserdata(L, (void*)signalKey);
      lua_rawget(L, LUA_REGISTRYINDEX);
      if (lua_istable(L, -1))
        {
          lua_pushlightuserdata(L, (void*)this);
          lua_rawget(L, -2);
          lua_remove(L, -2);
          if (lua_isfunction(L, -1))
            {
              for (int i=1; i<types.size(); i++)
                luaQ_p_push_retval(L, args[i], types[i]);
              bool oldRflag = d->rflag;
              d->rflag = true;
              locker.unlock();
              if (! oldRflag)
                emit d->q->stateChanged(QtLuaEngine::Running);
              if (lua_pcall(L, types.size()-1, 0, 0))
                if (!d || d->printErrors)
                  luaQ_print(L, 1);
            }
        }
      lua_settop(L, base);
    }
  else if (d)
    {
      // prepare queued signals
      QtLuaQueuedSignal q;
      q.sender = sender;
      q.receiver = this;
      q.delsignal = 0;
      for (int i=1; i<types.size(); i++)
        {
          int type = v_to_type(types[i]);
          const QMetaObject *mo = v_to_metaobject(types[i]);
          if (type == qMetaTypeId<QVariant>())
            q.args += *static_cast<QVariant*>(args[i]);
          else if (type)
            q.args += QVariant(type, args[i]);
          else if (mo) {
            QObjectPointer p = *static_cast<QObject**>(args[i]);
            q.args += qVariantFromValue(p);
          } else
            q.args += QVariant();
        }
      // queue signals for later invocation
      if (d->queuedSignals.isEmpty())
        queueSignal = true;
      d->queuedSignals += q;
      locker.unlock();
    }
  if (queueSignal)
    d->emitQueueSignal();
}


void
QtLuaEngine::Private::disconnectAllSignals()
{
  Receiver *r;
  foreach(QObject *obj, children())
    if ((obj) && (r = qobject_cast<Receiver*>(obj)))
      r->disconnect();
  QMutexLocker locker(&mutex);
  queuedSignals.clear();
}


bool 
QtLuaEngine::Private::processQueuedSignals(QMutexLocker &locker)
{
  bool processed = false;
  QList<QtLuaQueuedSignal> qs = queuedSignals;
  queuedSignals.clear();
  while (qs.size() > 0)
    {
      QtLuaQueuedSignal q = qs.takeFirst();
      if (q.delsignal)
        {
          // delete closure associated with this signal
          lua_pushlightuserdata(L, (void*)signalKey);
          lua_rawget(L, LUA_REGISTRYINDEX);
          if (lua_istable(L, -1))
            {
              lua_pushlightuserdata(L, (void*)this);
              lua_pushnil(L);
              lua_rawset(L, -3);
            }
          lua_pop(L, 1);
        }
      else if (q.sender && q.receiver && !pauseLoop && !unwindStack)
        {
          bool stateChanged = ! rflag;
          rflag = true;
          locker.unlock();
          if (stateChanged)
            emitStateChanged(QtLuaEngine::Running);
          // call closure
          int base = lua_gettop(L);
          lua_pushcfunction(L, lua_error_handler);
          lua_pushlightuserdata(L, (void*)signalKey);
          lua_rawget(L, LUA_REGISTRYINDEX);
          if (lua_istable(L, -1))
            {
              QtLuaEngine::Receiver *receiver = q.receiver;
              lua_pushlightuserdata(L, (void*)receiver);
              lua_rawget(L, -2);
              lua_remove(L, -2);
              if (lua_isfunction(L, -1))
                {
                  processed = true;
                  for (int i=0; i<q.args.size(); i++)
                    luaQ_pushqt(L, q.args[i]);
                  if (lua_pcall(L, q.args.size(), 0, -2-q.args.size()))
                    if (printErrors)
                      luaQ_print(L, 1);
                }
            }
          lua_settop(L, base);
          locker.relock();
        }
    }
  return processed;
}


/*! This function processes all pending events in the running thread.
  It also processed queued signals connected to a lua function or closure.
  This implies that these lua functions or closure are called from
  within the luaQ_doevents function. 
  The optional flag \a wait indicates if one should wait 
  until receiving at least one event. */

void 
luaQ_doevents(lua_State *L, bool wait)
{
  QtLuaEngine::Private *d = luaQ_private(L);
  QMutexLocker locker(&d->mutex);
  QEventLoop *saved_loop = d->pauseLoop;
  bool saved_rflag = d->rflag;
  // process events (state unchanged. should we pause on wait?)
  QEventLoop::ProcessEventsFlags flags = QEventLoop::AllEvents;
  if (wait && d->queuedSignals.isEmpty())
    flags |= QEventLoop::WaitForMoreEvents;
  locker.unlock();
  QCoreApplication::processEvents(flags);
  locker.relock();
  // run signals (possibly going into Running state)
  d->pauseLoop = 0;
  if (saved_loop)
    d->rflag = false;
  while(d->processQueuedSignals(locker))
    {
      locker.unlock();
      QCoreApplication::processEvents();
      locker.relock();
    }
  // restore initial state
  bool new_rflag = d->rflag;
  d->rflag = saved_rflag;
  d->pauseLoop = saved_loop;
  locker.unlock();
  if (saved_loop && new_rflag)
    emit d->emitStateChanged(QtLuaEngine::Paused);
  else if (!saved_rflag && new_rflag)
    emit d->emitStateChanged(QtLuaEngine::Ready);
}


void 
luaQ_doevents(lua_State *L)
{
  return luaQ_doevents(L, false);
}


void
luaQ_pause(lua_State *L)
{
  QtLuaEngine::Private *d = luaQ_private(L);
  d->emitQueueSignal();
  lua_pushliteral(L, "pause");
  QMutexLocker locker(&d->mutex);
  bool stateChanged = lua_pause_sub(L, d, locker);
  locker.unlock();
  if (stateChanged)
    d->emitStateChanged(QtLuaEngine::Running);
  lua_pop(L, 1);
  if (d && !d->unwindStack)
    return;
  lua_pushliteral(L, "stop");
  lua_error(L);
}


void
luaQ_resume(lua_State *L, bool nocontinue)
{
  QtLuaEngine::Private *d = luaQ_private(L);
  QMutexLocker locker(&d->mutex);
  if (d->pauseLoop)
    return;
  if (nocontinue)
    d->unwindStack = true;
  else
    d->resumeFlag = true;
}



/* create a receiver2 (receiver2, not receiver!) */

static int
luaQ_p_create_receiver(lua_State *L)
{
  QtLuaEngine::Private *d = luaQ_private(L);
  QtLuaEngine::Receiver *r = new QtLuaEngine::Receiver2(d->q);
  luaQ_pushqt(L, qVariantFromValue<void*>(static_cast<void*>(r)));
  return 1;
}


/*! Connects signal \a sig of Qt object \a obj to the 
  function located at stack position \a findex.
  Returns false if the signal does not exist. */

bool
luaQ_connect(lua_State *L, QObject *obj, 
             const char *sig, int findex, bool direct)
{
  bool success = false;
  QtLuaEngine::Private *d = luaQ_private(L);
  if (! lua_isfunction(L, findex))
    luaL_error(L, "luaQ_connect: function expected");
  // create receiver
  lua_pushcfunction(L, luaQ_p_create_receiver);
  luaQ_call(L, 0, 1, d->q);
  QVariant *vp = luaQ_toqvariantp(L, -1);
  void *v = (vp) ? qVariantValue<void*>(luaQ_toqvariant(L, -1)) : 0;
  QtLuaEngine::Receiver *r = static_cast<QtLuaEngine::Receiver*>(v);
  lua_pop(L, 1);
  if (! r)
    luaL_error(L, "luaQ_connect: cannot create receiver");
  // push closure
  lua_pushvalue(L, findex);
  lua_pushlightuserdata(L, (void*)signalKey);
  lua_rawget(L, LUA_REGISTRYINDEX);
  if (lua_istable(L, -1))
    {
      lua_pushlightuserdata(L, (void*)r);
      // ..stack: function signaltable receiveraddress
      lua_pushvalue(L, -3);
      lua_rawset(L, -3);
      success = r->connect(obj, sig, direct);
      if (! success)
        r->deleteLater();
    }
  lua_pop(L, 2);
  return success;
}


/*! Disconnects the connection matching the provided arguments.
  Arguments \a sig of \a findex can be zero to indicate
  that any signature or any closure should match. 
  Returns a boolean indicating if any such signals were found. */

bool
luaQ_disconnect(lua_State *L, QObject *obj, const char *sig, int findex)
{
  QtLuaEngine::Private *d = luaQ_private(L);
  if (sig && sig[0] == '0' + QSIGNAL_CODE)
    sig = sig + 1;
  else if (sig && sig[0] >= '0' && sig[0] <= '3')
    return false;
  if (findex && !lua_isfunction(L, findex))
    return false;
  QByteArray nsig = QMetaObject::normalizedSignature(sig);
  nsig.prepend('0' + QSIGNAL_CODE);
  lua_pushlightuserdata(L, (void*)signalKey);
  lua_rawget(L, LUA_REGISTRYINDEX);
  if (findex)
    lua_pushvalue(L, findex);
  else 
    lua_pushnil(L);
  // ..stack: function sigtable ...
  // select receviers
  QList<QtLuaEngine::Receiver*> chosen;
  QMutexLocker locker(&d->mutex);
  QtLuaEngine::Receiver *r;
  foreach(QObject *o, d->q->children())
    if ((r = qobject_cast<QtLuaEngine::Receiver*>(o)))
      {
        if (obj && r->sender != obj)
          continue;
        if (sig && r->signature != nsig)
          continue;
        if (findex && lua_istable(L, -2))
          {
            lua_pushlightuserdata(L, (void*)r);
            lua_rawget(L, -3);
            // ..stack: sigfunc function sigtable
            bool eq = lua_equal(L, -1, -2);
            lua_pop(L, 1);
            if (! eq)
              continue;
          }
        chosen += r;
      }
  // now disconnect and destroy the chosen receivers
  locker.unlock();
  foreach(r, chosen)
    {
      r->disconnect();
      lua_pushlightuserdata(L, (void*)r);
      lua_pushnil(L);
      lua_rawset(L, -4);
      r->deleteLater();
    }
  // return
  lua_pop(L, 2);
  return chosen.size() > 0;
}




// ========================================
// MOC

#include "qtluaengine.moc"




/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


