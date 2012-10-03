// -*- C++ -*-

#include "qluaconf.h"

#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#if HAVE_IO_H
# include <io.h>
#endif
#if HAVE_UNISTD_H
# include <unistd.h>
#endif

#include <QApplication>
#include <QDateTime>
#include <QDebug>
#include <QFile>
#include <QFileInfo>
#include <QFileOpenEvent>
#include <QFileDialog>
#include <QIcon>
#include <QList>
#include <QMessageBox>
#include <QMutex>
#include <QMutexLocker>
#include <QMap>
#include <QRegExp>
#include <QSet>
#include <QSettings>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QThread>

#include "lua.h"
#include "lauxlib.h"
#include "qtluaengine.h"
#include "qtluautils.h"

#include "qluaapplication.h"
#include "qluaconsole.h"





// ------- private class


struct QLuaApplication::Private : public QObject
{
  Q_OBJECT

public:

  struct Thread;

  const char * programName;
  QByteArray   programNameData;

  QLuaApplication *theApp;
  QLuaConsole *theConsole;
  QtLuaEngine *theEngine;
  Thread *theThread;
  
  int savedArgc;
  char **savedArgv;
  int  ttyEofCount;
  bool ttyEofReceived;
  bool ttyPauseReceived;
  bool interactionStarted;
  bool argumentsDone;
  bool closingDown;
  bool interactive;
  bool accepting;
  bool oneThread;
  bool ttyConsole;
  bool forceVersion;
  double elapsed;
  QDateTime startTime;
  QString aboutMessage;
  QByteArray luaPrompt;
  QByteArray luaPrompt2;
  QByteArray luaInput;
  QList<QObjectPointer> savedNamedObjects;
  QStringList filesToOpen;

  ~Private();
  Private(QLuaApplication *q);

  int printLuaVersion();
  int printUsage();
  int printMessage(int status, const char *fmt, ...);
  int printBadOption(const char *option);
  int doCall(struct lua_State *L, int nargs);
  int doLibrary(struct lua_State *L, const char *s);
  int doString(struct lua_State *L, const char *s);
  int doScript(struct lua_State *L, int argc, char **argv, int argn);
  int processArguments(int argc, char **argv);
  void acceptInput(bool clear);
  bool runCommand(QByteArray cmd, bool tty);
public slots:
  void start();
  void stateChanged(int state);
  void consoleBreak();
  void ttyInput(QByteArray ba);
  void ttyEndOfFile();
};


struct QLuaApplication::Private::Thread : public QThread
{
  Q_OBJECT
public:
  QLuaApplication::Private *d;
  QtLuaEngine *engine;
  QEventLoop *loop;
  QMutex mutex;
  Thread(QLuaApplication::Private *d);

  void preRun();
  void postRun();
  void run();
  void quit();
signals:
  void restart();
};


QLuaApplication::Private::Thread::Thread(QLuaApplication::Private *d) 
  : QThread(d->theApp), d(d), engine(0), loop(0)
{
  connect(this,SIGNAL(restart()),d,SLOT(start()),Qt::QueuedConnection);
}


void
QLuaApplication::Private::Thread::preRun()
{
  Q_ASSERT(!engine);
  Q_ASSERT(d->theConsole);
  engine = new QtLuaEngine;
  d->theEngine = engine;
  d->theConsole->setQtLuaEngine(engine, d->oneThread);
  engine->nameObject(d->theApp, "qApp");
  engine->nameObject(d->theEngine, "qEngine");
  engine->nameObject(d->theConsole, "qConsole");
  connect(engine, SIGNAL(stateChanged(int)),
          d, SLOT(stateChanged(int)), 
          Qt::QueuedConnection);
  // Advertise the new engine
  emit d->theApp->newEngine(engine);
  // Causes the emission of the first stateChanged() message.
  QtLuaLocker lock(engine); 
  lock.setRunning(); 
}


void
QLuaApplication::Private::Thread::run()
{
  QMutexLocker locker(&mutex);
  preRun();
  QEventLoop theloop;
  loop = &theloop;
  locker.unlock();
  loop->exec();
  locker.relock();
  loop = 0;
  postRun();
}


void
QLuaApplication::Private::Thread::quit()
{
  QMutexLocker locker(&mutex);
  if (loop)         // Unlike QThread::quit()
    loop->exit(0);  // this kill only the outer loop
}


void
QLuaApplication::Private::Thread::postRun()
{
  Q_ASSERT(engine);
  Q_ASSERT(d->theConsole);
  d->theConsole->setQtLuaEngine(0);
  d->theConsole->drainOutput();
  d->theEngine = 0;
  if (d->oneThread)
    engine->deleteLater();
  else
    delete engine;
  engine = 0;
  if (! d->closingDown)
    {
      fflush(stdout);
      fflush(stderr);
      fprintf(stderr,"\n------------ restarting lua ------------\n\n");
      fflush(stderr);
    }
  emit restart();
}


QLuaApplication::Private::~Private()
{
  closingDown = true;
  if (theEngine)
    {
      theEngine->setPrintResults(false);
      theEngine->setPrintErrors(false);
      theEngine->setPauseOnError(false);
    }
  if (theThread && theThread->isRunning())
    {
      Q_ASSERT(!oneThread);
      theEngine->resume(true);
      theEngine->stop(true);
      theThread->quit();
      // Wait for termination of the thread
      // while keeping the event loop active.
      while (theThread->isRunning())
        theApp->processEvents(QEventLoop::ExcludeUserInputEvents, 1000);
    }
  if (theEngine)
    {
      Q_ASSERT(oneThread);
      theThread->postRun();
    }
  Q_ASSERT(!theEngine);
}


QLuaApplication::Private::Private(QLuaApplication *q)
  : QObject(q),
    theApp(q), 
    theConsole(0), 
    theEngine(0), 
    theThread(new Thread(this)),
    savedArgc(0),
    savedArgv(0),
    ttyEofCount(0),
    ttyEofReceived(false),
    ttyPauseReceived(false),
    interactionStarted(false),
    argumentsDone(false),
    closingDown(false),
    interactive(false),
    accepting(false),
    oneThread(false),
    ttyConsole(false),
    forceVersion(false),
    elapsed(0),
    luaPrompt("> "),
    luaPrompt2(">> ")
{
  programName = "qlua";
  aboutMessage = "<html>"
    "<b>QLua " QLUA_VERSION ":</b>"
    "<ol><li>A Lua scripting engine for Qt4.</li>"
    "<li>Qt bindings for Lua.</li></ol><p>"
    "Copyright &copy; 2008-&middot; NEC Laboratories America (L. Bottou).<p>"
    "The QLua specific source code is distributed under a BSD license.&nbsp;"
    "See the file <verb>COPYRIGHT.txt</verb> for details.&nbsp;"
    "It does not include the Qt4 library which is available from "
    "TrollTech under various licenses."
    "</html>";
}


void
QLuaApplication::Private::start()
{
  Q_ASSERT(theThread && !theEngine);
  theThread->wait();
  if (closingDown)
    return;
  if (oneThread)
    theThread->preRun();
  else
    theThread->start();
}





// -------- private


int
QLuaApplication::Private::printLuaVersion()
{
  fprintf(stderr, "%s\n", LUA_VERSION "  " LUA_COPYRIGHT);
  return 0;
}


int
QLuaApplication::Private::printUsage()
{
  fflush(stdout);
  fflush(stderr);
  fprintf(stderr,
          ("Usage: %s [options] [script <scriptargs>]\n"
           "The lua options are:\n"
           "  -e stat       execute string 'stat'\n"
           "  -l name       require library 'name'\n"
           "  -i            enter interactive mode after executing 'script'\n"
           "  -v            show version information\n"
           "  --            stop handling options\n"
           "  -             execute stdin and stop handling options\n"
           "Specific options for qlua:\n"
           "  -ide          run the IDE with the last style\n"
           "  -ide=style    run the IDE and specify a style (sdi,mdi,tab)\n"
           "  -onethread    run lua in the main thread\n"
           "  -nographics   disable all the graphical capabilities\n"
           "  -style s      set the application gui style 's'\n"
           "  -session s    restore gui session 's'\n"
#ifdef Q_WS_X11
           "  -display d    set the X display (default is $DISPLAY)\n"
           "  -geometry g   set the main window size and position\n"
           "  -title s      set the main window title to 's'\n"
#endif
           "  ...           see the Qt documentation for more options\n"),
          programName); 
  return 0;
}


int
QLuaApplication::Private::printMessage(int status, const char *fmt, ...)
{
  fflush(stdout);
  fflush(stderr);
  if (programName)
    fprintf(stderr,"%s: ", programName);
  if (!fmt)
    fmt = "unprintable error";
  QString message;
  va_list ap;
  va_start(ap, fmt);
  message = message.vsprintf(fmt, ap);
  va_end(ap);
  fprintf(stderr, "%s\n", message.toLocal8Bit().constData());
  return status;
}


int
QLuaApplication::Private::printBadOption(const char *option)
{
  printMessage(0, "Option '%s' is not recognized.\n", option);
  printMessage(0, "Type 'qlua -h' for usage information\n");
  return -1;
}



int 
QLuaApplication::Private::doCall(struct lua_State *L, int nargs)
{
  int status;
  int base = lua_gettop(L) - nargs;
  lua_pushcfunction(L, luaQ_traceback);
  lua_insert(L, base);
  status = luaQ_pcall(L, nargs, 0, base, theEngine);
  lua_remove(L, base);
  if (status)
    lua_gc(L, LUA_GCCOLLECT, 0);
  return status;
}


int 
QLuaApplication::Private::doLibrary(struct lua_State *L, const char *s)
{
  lua_getglobal(L, "require");
  if (! lua_isfunction(L, -1))
    return printMessage(-1, "global 'require' is not a function");
  lua_pushstring(L, s);
  int status = doCall(L, 1);
  if (status)
    return printMessage(status, lua_tostring(L, -1));
  return status;
}


int 
QLuaApplication::Private::doString(struct lua_State *L,
                                   const char *s)
{
  int status = luaL_loadstring(L, s);
  if (status)
    return printMessage(status, "%s", lua_tostring(L, -1));
  if ((status = doCall(L, 0)))
    return printMessage(status, "%s", lua_tostring(L, -1));
  return status;
}


int 
QLuaApplication::Private::doScript(struct lua_State *L, 
                                   int argc, char **argv, int argn)
{
  int i;
  int status;
  if (! lua_checkstack(L, argc+4))
    return printMessage(-1, "stack overflow (too many arguments)");
  Q_ASSERT(argn<=argc);
  for (i=argn+1; i<argc; i++)
    lua_pushstring(L, argv[i]);
  lua_createtable(L, argc-argn, argn+1);
  for (i=0; i<=argn; i++) {
    lua_pushstring(L, (i<argc) ? argv[i] : "-");
    lua_rawseti(L, -2, i-argn);
  }
  for (i=argn+1; i<argc; i++) {
    lua_pushvalue(L, i-argc-1);
    lua_rawseti(L, -2, i-argn);
  }
  lua_setglobal(L, "arg");
  const char *s = 0;
  if (argn<argc && strcmp(argv[argn],"-") && strcmp(argv[argn],"--"))
    s = argv[argn];
  if ((status = luaL_loadfile(L, s)))
    return printMessage(status, "%s", lua_tostring(L, -1));
  int narg = (argn<argc) ? argc-argn-1 : 0;
  lua_insert(L, -narg-1);
  if ((status = doCall(L, narg)))
    return printMessage(status, "%s", lua_tostring(L, -1));
  return status;
}


static int
no_call(lua_State *L)
{
  luaL_error(L, "This class prevents lua to call this method");
  return 0;
}


static int
hook_qluaconsole(lua_State *L)
{
  // ..stack: metatable
  lua_getfield(L, -1, "__metatable");
  // ..stack: metaclass
  lua_pushcfunction(L, no_call);
  lua_setfield(L, -2, "deleteLater");
  lua_pushcfunction(L, no_call);
  lua_setfield(L, -2, "deleteLater(QObject*)");
  return 0;
}


int
QLuaApplication::Private::processArguments(int argc, char **argv)
{
  bool has_e = false;
  bool has_v = false;
  bool stdinmode = false;

  // Obtain and lock lua
  QtLuaLocker lua(theEngine);
  
  // Good time to limit access to QtLuaConsole
  lua_pushcfunction(lua, hook_qluaconsole);
  luaQ_pushmeta(lua, &QLuaConsole::staticMetaObject);
  lua_call(lua, 1, 0);

  // parse lua argument 
  int argn = 1;
  int status;
  while (argn < argc)
    {
      const char *a;
      const char *s = argv[argn];
      if (s[0] != '-')
        break;
      if (s[1] == 0)
        break;
      argn += 1;
      if (s[1] == '-' && s[2] == 0)
        break;
      switch(s[1])
        {
        case '-':
          if (s[2]) 
            return printBadOption(s);
          break;
        case 'i':
          if (!strcmp(s, "-ide") || !strncmp(s, "-ide=", 5))
            break;
          else if (s[2]) 
            return printBadOption(s);
          interactive = ttyConsole = true;
          theConsole->setCtrlCHandler(QLuaConsole::ctrlCBreak);
          theConsole->setPrintCapturedOutput(true);
          break;
        case 'v':
          if (s[2]) 
            return printBadOption(s);
          has_v = true;
          break;
        case 'e':
          has_e = true;
          a = s + 2;
          if (a[0]==0 && argn < argc)
            a = argv[argn++];
          lua.setRunning();
          if (a && a[0])
            if ((status = doString(lua, a)))
              return status;
          break;
        case 'l':
          a = s + 2;
          if (a[0]==0 && argn < argc)
            a = argv[argn++];
          lua.setRunning();
          if (a && a[0])
            if ((status = doLibrary(lua, a)))
              return status;
          break;
        case 'h':
          if (s[2])
            return printBadOption(s);
          return printUsage();
          break;
        case 'n':
        case 'o':
        default:
          if (strcmp(s, "-nographics") &&
              strcmp(s, "-onethread") )
            return printBadOption(s);
          break;
        }
    }
  // script mode?
  if (argn>=argc && !has_e && !has_v)
    {
      int c = EOF;
#if HAVE_ISATTY
      bool stdin_is_tty = isatty(0);
#elif defined(WIN32)
      bool stdin_is_tty = _isatty(_fileno(stdin));
#else
      bool stdin_is_tty = true;
#endif
      if (stdin_is_tty)
        interactive = ttyConsole = true;
      else if ((c = fgetc(stdin)) != EOF)
        stdinmode = true;
      if (stdinmode)
        ungetc(c, stdin);
    }
  // handle script
  if (argn < argc)
    if ((status = doScript(lua, argc, argv, argn)))
      return status;
  // handle stdin
  if (stdinmode)
    if ((status = doScript(lua, argc, argv, argc)))
      return status;
  // run interactive if there are toplevel windows
  foreach(QWidget *w, QApplication::topLevelWidgets())
    if (w && w->isVisible() && w->windowType() != Qt::Desktop)
      interactive = true;
  // do we need to print the version?
  forceVersion = has_v;
  if (has_v && !interactive)
    printLuaVersion();
  return 0;
}




// -------- interaction


void 
QLuaApplication::Private::stateChanged(int state)
{
  // safeguard
  if (closingDown || !theEngine)
    return;
  // timing
  if (state == QtLuaEngine::Running)
    {
      if (ttyPauseReceived)
        {
          ttyPauseReceived = false;
          theConsole->abortReadLine();
        }
      startTime = QDateTime::currentDateTime();
    }
  else
    {
      QDateTime now = QDateTime::currentDateTime();
      int secs = startTime.secsTo(now);
      int msecs = startTime.time().msecsTo(now.time()) % 1000;
      elapsed += secs + (msecs * 0.001);
    }
  // dealing with pauses
  if (state == QtLuaEngine::Paused)
    {
      if (theEngine->isPaused())
        {
          if (! interactionStarted)
            {
              theEngine->resume(false);
            }
          else if (ttyConsole)
            {
              QByteArray prompt = "[Pause -- press enter to continue] ";
              ttyPauseReceived = true;
              theConsole->readLine(prompt);
            }
        }
    }
  // accepting new commands
  if (state == QtLuaEngine::Ready)
    {
      while (! savedNamedObjects.isEmpty())
        {
          QObjectPointer ptr = savedNamedObjects.takeFirst();
          if (ptr) theEngine->nameObject(ptr);
        }
      if (! argumentsDone)
        {
          argumentsDone = true;
          theEngine->setPrintResults(false);
          theApp->setQuitOnLastWindowClosed(false);
          int status = processArguments(savedArgc, savedArgv);
          if (status && !interactive)
            { theApp->exit(status = EXIT_FAILURE); return; }
          else if (!interactive)
            { theApp->exit(status = EXIT_SUCCESS); return; }
        }
      // go in interactive mode
      if (!interactionStarted)  
        {
          interactionStarted = true;
          theApp->setupConsoleOutput();
          theApp->setQuitOnLastWindowClosed(!ttyConsole);
          bool capture = theConsole->captureOutput();
          theEngine->setPrintResults(ttyConsole || capture);
          if (forceVersion || theEngine->printResults())
            printLuaVersion();
        }
      // accept fresh commands
      if (theEngine->isReady())
        acceptInput(false);
    }
}


void 
QLuaApplication::Private::consoleBreak()
{
  if (! theEngine->stop(true) )
    if (theEngine->isReady())
      acceptInput(true);
}


void 
QLuaApplication::Private::acceptInput(bool clear)
{
  // update the prompt
  QtLuaLocker lua(theEngine, 0);
  struct lua_State *L = lua;
  if (L)
    {
      struct lua_State *L = lua;
      lua_getfield(L, LUA_GLOBALSINDEX, "_PROMPT");
      lua_getfield(L, LUA_GLOBALSINDEX, "_PROMPT2");
      luaPrompt = lua_isstring(L,-2) ? lua_tostring(L, -2) : "> ";
      luaPrompt2 = lua_isstring(L,-1) ? lua_tostring(L, -1) : ">> ";
      lua_pop(L, 2);
    }
  // clear console
  theConsole->drainOutput();
  // do not quit to abruptly
  theApp->setQuitOnLastWindowClosed(!ttyConsole);
  // signal acceptance
  accepting = true;
  emit theApp->acceptingCommands(true);
  // read line from the tty
  if (ttyConsole && theApp->isAcceptingCommands())
    {
      if (clear) {
        luaInput.clear();
        theConsole->abortReadLine();
      } else
        theConsole->redisplayReadLine();
      theConsole->setCtrlCHandler(QLuaConsole::ctrlCBreak);
      theConsole->readLine(luaPrompt.constData());
    }
}


void 
QLuaApplication::Private::ttyInput(QByteArray ba)
{
  // are we in a pause?
  if (ttyPauseReceived)
    {
      ttyPauseReceived = false;
      theEngine->resume(false);
      return;
    }
  // are we quitting?
  if (ttyEofReceived)
    {
      ttyEofReceived = false;
      if ((ba[0]=='y' || ba[0]=='Y'))
        {
          if (! theApp->close())
            acceptInput(true);
        }
      else if (ba.size()==0 || ba[0]=='n' || ba[0]=='N') 
        {
          ttyEofCount = 0;
          acceptInput(true);
        }
      else
        ttyEndOfFile();
      return;
    }
  // append to the current input
  if (! luaInput.size())
    {
      luaInput = ba;
      if (ba.size() > 0 && ba[0] == '=')
        luaInput = QByteArray("return ") + (ba.constData() + 1);
    }
  else
    {
      luaInput += '\n';
      luaInput += ba;
    }
  // determine if line is complete
  QtLuaLocker lua(theEngine, 1000);
  struct lua_State *L = lua;
  const char *data = luaInput.constData();
  int status = (L) ? 0 : 1;
  if (! status)
    {
      status = luaL_loadbuffer(L, data, luaInput.size(), "=stdin");
      if (status == LUA_ERRSYNTAX) 
        {
          size_t lmsg;
          const char *msg = lua_tolstring(L, -1, &lmsg);
          const char *tp = msg + lmsg - (sizeof(LUA_QL("<eof>")) - 1);
          status = (strstr(msg, LUA_QL("<eof>")) == tp);
        }
      lua_pop(L, 1);
    }
  // action
  if (status)
    {
      theConsole->readLine(luaPrompt2.constData());
    }
  else
    {
      QByteArray cmd = luaInput;
      luaInput.clear();
      theConsole->addToHistory(cmd);
      if (cmd.simplified().isEmpty())
        acceptInput(true);
      else
        runCommand(cmd, false);
    }
}


bool
QLuaApplication::Private::runCommand(QByteArray cmd, bool ttyEcho)
{
  if (! theEngine)
    return false;
  QtLuaLocker lua(theEngine, 1000);
  if (!lua || !theEngine->isReady())
    return false;
  // ready to go
  if (ttyConsole)
    {
      luaInput.clear();
      theConsole->abortReadLine();
      if (ttyEcho)
        {
          bool captured = theConsole->captureOutput();
          QByteArray s = cmd;
          s.replace("\n", "\n" + luaPrompt2);
          s.prepend("\r" + luaPrompt);
          fflush(stdout);
          theConsole->setCaptureOutput(false);
          fprintf(stdout,"%s\n", s.constData());
          fflush(stdout);
          theConsole->setCaptureOutput(captured);
        }
    }
  elapsed = 0;
  accepting = false;
  emit theApp->acceptingCommands(false);
  emit theApp->luaCommandEcho(luaPrompt, luaPrompt2, cmd);
  return theEngine->eval(cmd, !oneThread);
}


void 
QLuaApplication::Private::ttyEndOfFile()
{
  if (++ttyEofCount > 8)
    {
      sendPostedEvents();
      theApp->quit();
    }
  else if (ttyConsole && interactionStarted)
    {
      accepting = false;
      emit theApp->acceptingCommands(false);
      luaInput.clear();
      QByteArray prompt = "Really quit [y/N]? ";
      ttyEofReceived = true;
      theConsole->abortReadLine();
      theConsole->readLine(prompt);
    }
}



// -------- application


QLuaApplication::~QLuaApplication()
{
  delete d;
}


static QString
capitalize(QString s)
{
  s = s.toLower();
  if (s.size() > 0)
    s[0] = s[0].toUpper();
  return s;
}


/*! Creates the application.
  Processes command line arguments directed to Qt
  and leaves the remaining command line arguments alone. */

QLuaApplication::QLuaApplication(int &argc, char **argv, 
                                 bool guiEnabled, bool onethread)
  : QApplication(argc, argv, guiEnabled),
    d(new Private(this))
{
  // one thread only
  d->oneThread = onethread;

  // extract program name
  QString cuteName = QFileInfo(applicationFilePath()).baseName();
  d->programNameData = cuteName.toLocal8Bit();
  d->programName = d->programNameData.constData();
  QRegExp re("^(mac(?=qlua)|win(?=qlua)|)(q?)(.*)", Qt::CaseInsensitive);
  cuteName = capitalize(cuteName);
  if (re.indexIn(cuteName) >= 0 && re.numCaptures() == 3)
    cuteName = capitalize(re.cap(2)) + capitalize(re.cap(3));

  // basic setup
  setApplicationName(cuteName);
  setOrganizationName(QLUA_ORG);
  setOrganizationDomain(LUA_DOMAIN);

#ifndef Q_WS_MAC
  if (guiEnabled && windowIcon().isNull())
    setWindowIcon(QIcon(":/qlua.png"));
#else
  extern void qt_mac_set_native_menubar(bool);
  extern void qt_mac_set_menubar_icons(bool);
# ifdef QT_MAC_USE_NATIVE_MENUBAR
  qt_mac_set_native_menubar(true);
# else
  qt_mac_set_native_menubar(false);
# endif
  qt_mac_set_menubar_icons(false);
#endif
  
  // create console
  //   It is important to create this first because
  //   the console code ensures that posix signals are
  //   processed from the console thread.
  d->theConsole = new QLuaConsole(this);
  connect(d->theConsole, SIGNAL(ttyBreak()),
          d, SLOT(consoleBreak()) );
  connect(d->theConsole, SIGNAL(ttyInput(QByteArray)),
          d, SLOT(ttyInput(QByteArray)) );
  connect(d->theConsole, SIGNAL(ttyEndOfFile()),
          d, SLOT(ttyEndOfFile()) );
  connect(d->theConsole, SIGNAL(consoleOutput(QByteArray)),
          this, SIGNAL(luaConsoleOutput(QByteArray)) );
}




// --- startup

/*! Start the lua engine and enter the qt main loop. 
  Arguments \a argc and \a argv replicate the
  behavior of the traditional \a "lua" command line
  program. */

int
QLuaApplication::main(int argc, char **argv)
{
  d->savedArgc = argc;
  d->savedArgv = argv;
  d->start();
  // Run qt main loop;
  // Command line argument will be processed when
  // we will receive signal readyState.
  return exec();
}





// ----- utilities


/*! Return the unique QLuaApplication instance. */

QLuaApplication*
QLuaApplication::instance()
{
  return qobject_cast<QLuaApplication*>(QCoreApplication::instance());
}


/*! Return the unique QConsole instance. */

QLuaConsole *
QLuaApplication::console()
{
  QLuaApplication *app = instance();
  return (app) ? app->d->theConsole : 0;
}


/*! Return the currently running QtLuaEngine.
  This can change or even be zero 
  when one calls \a restart(). */

QtLuaEngine *
QLuaApplication::engine()
{
  QLuaApplication *app = instance();
  return (app) ? app->d->theEngine : 0;
}


/*! Tells whether the application is accepting
  interactive commands with \a runCommand. */

bool 
QLuaApplication::isAcceptingCommands() const
{
  return d->accepting;
}


/*! Tells if the application is running a close sequence.  
  This is not the same as \a QCoreApplication::closingDown(). */

bool 
QLuaApplication::isClosingDown() const
{
  return d->closingDown;
}




// ----- signals

/*! \signal newEngine()
  Invoked when a new lua engine is created after \a restart().
 */

/*! \signal acceptingCommands(bool)
  Indicates whether the application is currently 
  accepting Lua commands using \a runCommand().
 */

/*! \signal luaCommandEcho(QByteArray ps1,QByteArray ps2,QByteArray cmd)
  Invoked when one calls \a runCommand.
  Arguments \a ps1 and \a ps2 are the primary and secondary lua prompts.
  Argument \a cmd is the command.
*/

/*! \signal luaConsoleOutput
  This signal relays the console \
  signal \a consoleOutput for convenience.
 */

/*! \signal anyoneHandlesConsoleOutput(bool&)
  Applications that need the console output should listen to
  this signal with a direct connection and set the boolean to \a true.
  This is used by \a setupConsoleOutput.
 */

/*! \signal fileOpenEvent()
  This is emitted when we receive a \a QFileOpenEvent.
  Use \a filesToOpen to obtain the filenames.
  This is a Macintosh thing... 
*/

/*! \signal windowShown(QWidget *window)
  This signal indicates that a new toplevel window is ready to be shown.
  Clients can capture this signal to implement window placement policies
  such as capturing the windows into a MDI environment.
  This feature works by intercepting events of type QEvent::Show. 
  Be careful as a window can be shown several times!
*/



// ----- actions


/*! Read application settings for key \a key.
   This function can be useful to lua programs. */

QVariant 
QLuaApplication::readSettings(QString key)
{
  QSettings s;
  return s.value(key);
}


/*! Write application settings for key \a key.
   This function can be useful to lua programs. */

void 
QLuaApplication::writeSettings(QString key, QVariant value)
{
  QSettings s;
  if (value.type() == QVariant::Invalid)
    s.remove(key);
  else
    s.setValue(key,value);
}


/*! Return files that were passed to the system 
  using QFileOpenEvent messages, then clears the list. */

QStringList 
QLuaApplication::filesToOpen()
{
  QStringList l = d->filesToOpen;
  d->filesToOpen.clear();
  return l;
}


/*! Return true if qlua option -nographics is in effect. */

bool 
QLuaApplication::runsWithoutGraphics() const
{
  return (QApplication::type() == QApplication::Tty);
}


/*! Return the number of seconds elapsed for 
  running the last command with \a runCommand.
  This is invalid while the command is running. */

double 
QLuaApplication::timeForLastCommand() const
{
  return d->elapsed;
}


bool 
QLuaApplication::event(QEvent *e)
{
  if (e->type() == QEvent::FileOpen)
    {
      QString f = static_cast<QFileOpenEvent*>(e)->file();
      d->filesToOpen.append(f);
      emit fileOpenEvent();
      return true;
    }
  else if (e->type() == QEvent::Close)
    {
      // carefully close all windows
      bool okay = true;
      QSet<QWidget*> closed;
      QWidgetList wl = topLevelWidgets();
      while (okay && wl.size())
        {
          QWidget *w = wl.takeFirst();
          if (w == 0 || !w->isVisible() ||
              w->windowType() == Qt::Desktop ||
              closed.contains(w) )
            continue;
          closed += w;
          okay = w->close();
          wl = topLevelWidgets();
        }
      // accept event on success
      e->setAccepted(okay);
      return true;
    }
  return QApplication::event(e);
}


bool 
QLuaApplication::notify(QObject *receiver, QEvent *event)
{
  // capture window polish events
  if (event->type() == QEvent::Show)
    if (receiver->isWidgetType())
      {
        QWidget *w = static_cast<QWidget*>(receiver);
        if (w->windowType() == Qt::Window)
#if QT_VERSION >= 0x040400 
          if (!w->testAttribute(Qt::WA_DontShowOnScreen))
#endif
            emit windowShown(w);
      }
  return QApplication::notify(receiver, event);
}


/*! Automatically determine the right state of the
  console properties \a captureOutput and \a printCapturedOutput
  on the basis of the answers to signal \a anyoneHandlesConsoleOutput
  and the presence of a console. This should be called whenever
  a user of the console output appears or disappears. */

void 
QLuaApplication::setupConsoleOutput()
{
  bool capture = false;
  emit anyoneHandlesConsoleOutput(capture);
  d->theConsole->setPrintCapturedOutput(d->ttyConsole);
  d->theConsole->setCaptureOutput(capture);
}


/*! Execute an interactive Lua command.
  This function could be called after 
  receiving \a acceptingCommands(true)
  for executing lua command entered by the user.
  It immediately returns \a false if predicate
  \a isAcceptingCommands() is not true.
  It causes the emission of signal \a luaCommandEcho
  and displays the command on the console when
  available. */

bool 
QLuaApplication::runCommand(QByteArray cmd)
{
  if (d->accepting)
    return d->runCommand(cmd, true);
  return false;
}


/*! Restart the interpreter afresh.
  When argument \a redoCmdLine is \a true,
  the command line arguments are re-executed
  by the new lua interpreter.
  Named objects that were defined in the
  lua engine are preserved in the new engine. */

void
QLuaApplication::restart(bool redoCmdLine)
{
  d->accepting = false;
  emit acceptingCommands(false);
  d->savedNamedObjects = d->theEngine->allNamedObjects();
  d->argumentsDone = !redoCmdLine;
  d->interactionStarted = false;
  // silence
  QMutexLocker locker(&d->theThread->mutex);
  QtLuaEngine *engine = d->theEngine;
  if (engine)
    {
      disconnect(engine, 0, 0, 0);
      engine->setPrintErrors(false);
      engine->setPrintResults(false);
      engine->setPauseOnError(false);
      engine->resume(true);
      engine->stop(true);
    }
  d->theEngine = 0;
  locker.unlock();
  // stop
  if (d->oneThread)
    d->theThread->postRun();
  else
    d->theThread->quit();
}


/*! \property QLuaApplication::aboutMessage
  Contains the message displayed by \a about().
  Substring \a "${APPNAME}" is replaced by the application name. */

QString 
QLuaApplication::aboutMessage() const
{
  QString an = applicationName();
  return d->aboutMessage.replace("${APPNAME}", an);
}


void 
QLuaApplication::setAboutMessage(QString m)
{
  d->aboutMessage = m;
}


/*! Display an dialog with the about message
  specified by property \a aboutMessage. */

void 
QLuaApplication::about()
{
  QWidget *w = qobject_cast<QWidget*>(sender());
  QString an = applicationName();
  QMessageBox::about(w, tr("About %1").arg(an), aboutMessage());
}


/*! Quit the application properly, 
  First sends a close event to the application.
  The default close event handler closes all
  windows and accepts the event in case of success.
  Returns true on success. */

bool
QLuaApplication::close()
{
  if (! d->closingDown)
    {
      d->closingDown = true;
      QCloseEvent ev;
      sendEvent(this, &ev);
      d->closingDown = ev.isAccepted();
      if (d->closingDown)
        {
          sendPostedEvents();
          QTimer::singleShot(0, this, SLOT(quit()));
        }
    }
  return d->closingDown;
}

  



#include "qluaapplication.moc"


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ( "\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" )
   End:
   ------------------------------------------------------------- */



