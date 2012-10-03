// -*- C++ -*-


#include "qluaconf.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <exception>
#include <ctype.h>

#if HAVE_ERRNO_H
# include <errno.h>
#endif
#if HAVE_FCNTL_H
# include <fcntl.h>
#endif
#if HAVE_IO_H
# include <io.h>
#endif
#if HAVE_PTHREAD_H
# include <pthread.h>
#endif
#if HAVE_SIGNAL_H
# include <signal.h>
#endif
#if HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#if HAVE_SYS_TIME_H
# include <sys/time.h>
#endif
#if HAVE_SYS_SELECT_H
# include <sys/select.h>
#endif
#if HAVE_UNISTD_H
# include <unistd.h>
#endif
#if HAVE_READLINE
# ifdef HAVE_RL_COMPLETION_MATCHES
#  include <readline/readline.h>
#  include <readline/history.h>
# else
#  undef HAVE_READLINE
# endif
#endif

#include <QtGlobal>
#include <QtAlgorithms>
#include <QCoreApplication>
#include <QDebug>
#include <QFileInfo>
#include <QMutex>
#include <QMutexLocker>
#include <QPointer>
#include <QSettings>
#include <QString>
#include <QStringList>
#include <QThread>
#include <QTimer>
#include <QWaitCondition>

#include "lua.h"
#include "lauxlib.h"

#include "qluaconsole.h"
#include "qtluaengine.h"
#include "qtluautils.h"


/* some compilers define getchar() which causes problems below */
#undef getchar


// ----------------------------------------
// Manipulating signals


typedef void (*signal_handler_t)(int);

#if APPLE

static void hold_signals(sigset_t*) { }
static void release_signals(sigset_t*) { }

static signal_handler_t
set_sigint_handler(signal_handler_t handler)
{
  return signal(SIGINT, handler);
}

#elif HAVE_SIGACTION

static void 
hold_signals(sigset_t *oset)
{
  sigset_t sset;
  sigemptyset(&sset);
# ifdef SIGINT
  sigaddset(&sset, SIGINT);
# endif
# ifdef SIGTSTP
  sigaddset(&sset, SIGTSTP);
# endif  
# ifdef SIGSTOP
  sigaddset(&sset, SIGSTOP);
# endif  
# ifdef SIGQUIT
  sigaddset(&sset, SIGQUIT);
# endif  
# ifdef SIGHUP
  sigaddset(&sset, SIGHUP);
# endif  
# ifdef SIGTERM
  sigaddset(&sset, SIGTERM);
# endif  
# ifdef SIGCONT
  sigaddset(&sset, SIGCONT);
# endif  
# ifdef SIGWINCH
  sigaddset(&sset, SIGWINCH);
# endif  
# if HAVE_PTHREAD_SIGMASK
  pthread_sigmask(SIG_BLOCK, &sset, oset);
# elif HAVE_SIGPROCMASK
  sigprocmask(SIG_BLOCK, &sset, oset);
# endif
}

static void 
release_signals(sigset_t *oset)
{
# if HAVE_PTHREAD_SIGMASK
  pthread_sigmask(SIG_SETMASK, oset, 0);
# elif HAVE_SIGPROCMASK
  sigprocmask(SIG_SETMASK, oset, 0);
# endif
}

template<typename V, typename P> 
static void assign_any_pointer(V* &var, P* ptr)
{
  var = static_cast<V*>(ptr);
}

static signal_handler_t
set_sigint_handler(signal_handler_t handler)
{
  struct sigaction act;
  struct sigaction oct;
  signal_handler_t ret;
  memset(&act, 0, sizeof(act));
  memset(&oct, 0, sizeof(act));
  assign_any_pointer(act.sa_handler, handler);
  sigemptyset(&act.sa_mask);
  act.sa_flags = 0;
  sigaction(SIGINT, &act, &oct);
  assign_any_pointer(ret, oct.sa_handler);
  return ret;
}

#elif HAVE_SIGNAL

typedef int sigset_t;
static void hold_signals(sigset_t*) { }
static void release_signals(sigset_t*) { }

static signal_handler_t
set_sigint_handler(signal_handler_t handler)
{
  return signal(SIGINT, handler);
}

#else // !HAVE_SIGACTION && !HAVE_SIGNAL

typedef int sigset_t;
static void hold_signals(sigset_t*) { }
static void release_signals(sigset_t*) { }

static signal_handler_t
set_sigint_handler(signal_handler_t handler)
{
  static signal_handler_t fake = 0;
  signal_handler_t old = fake;
  fake = handler;
  return old;
}


#endif // !HAVE_SIGACTION && !HAVE_SIGNAL





// ----------------------------------------
// Manipulating file descriptors


static int
wait_for_input(int nfds, int *fds, bool wait=true)
{
  int i;
  int maxfd = 0;
  fd_set fdset;
  FD_ZERO(&fdset);
  for (i=0; i<nfds; i++)
    if (fds[i] >= 0)
      {
        maxfd = qMax(maxfd, fds[i]);
        FD_SET(fds[i], &fdset);
      }
  // Use the good old selection function
  struct timeval tv;
  tv.tv_sec = 0;
  tv.tv_usec = 0;
  int status = select(maxfd+1, &fdset, 0, 0, ((wait) ? 0 : &tv));
  if (status > 0)
    for (i=0; i<nfds; i++)
      if (fds[i] >= 0 && FD_ISSET(fds[i], &fdset))
        return fds[i];
  return -1;
}





// ----------------------------------------
// QLuaConsole::Private


enum Command { 
  NoCmd, 
  KillCmd, 
  ReadlineCmd, 
  AbortCmd,
  RedisplayCmd,
  BreakCmd,
  DrainCmd,
  HandlerCmd
};


struct QLuaConsole::Private : public QThread
{
  Q_OBJECT

public:
  
  QLuaConsole *q;
  QMutex mutex;
  QWaitCondition pulse;
  QWaitCondition drain;
  int  throttleCount;
  bool throttleActive;

  sigset_t savedSigSet;
  int stdoutPipe[2];
  int commandPipe[2];

  bool killConsole;
  bool captureOutput;
  bool printCapturedOutput;
  bool redirected;

  FILE *trueStdout;
  FILE *trueStderr;

  QByteArray prompt;
  QPointer<QtLuaEngine> lua;
  bool breakStopsLua;
  CtrlCHandler ctrlCHandler;
  
  Private(QLuaConsole *parent);
  ~Private();
  void run();
  int  getchar();
  void readline();
  void copyout(bool throttle=true);
  void sigint();
  void command(enum Command);
  void redirect(bool);
  void sethandler();

public slots:
  void slotConsoleOutput(QByteArray ba);
  void slotCommandNoCmd();
signals:
  void sigConsoleOutput(QByteArray ba);
};


static QLuaConsole::Private *console;



// ----------------------------------------
// Reading from the tty


enum RttyStatus { RttyOk, RttyEof, RttyAbort };
static enum RttyStatus rtty_status = RttyOk;

static int
rtty_getchar(FILE*)
{
  int c;
  if (console)
    c = console->getchar();
  else 
    c = getchar();
  return c;
}


#if HAVE_READLINE

static const char *
rtty_keywords[] = 
  {
    "and", "break", "do", "else", "elseif", 
    "end", "false", "for", "function",
    "if", "in", "local", "nil", "not", 
    "or", "repeat", "return", "then",
    "true", "until", "while", 0
  };

static int 
rtty_lex(const char *s, int end, int &q)
{
  int state = -1;
  int n = 0;
  int p = 0;
  while (p < end)
    {
      switch(state)
        {
        default:
        case -1: // misc
          if (isalpha(s[p]) || s[p]=='_') {
            q = p; state = -2; 
          } else if (s[p]=='\'') {
            q = s[p]; n = 0; state = -3; 
          } else if (s[p]=='\"') {
            q = s[p]; n = 0; state = -3;
          } else if (s[p]=='[') {
            n = 0; state = -3;
            const char *t = s + p + 1;
            while (*t == '=')
              t += 1;
            if (*t == '[')
              n = t - s - p;
            else
              state = -1;
          } else if (s[p]=='-' && s[p+1]=='-') {
            n = 0; state = -4; 
            if (s[p+2]=='[') {
              const char *t = s + p + 3;
              while (*t == '=')
                t += 1;
              if (*t == '[')
                n = t - s - p - 2;
            }
          }
          break;
        case -2: // identifier
          if (s[p] == '(' || s[p] == '{') {
            state = -5;
          } else if (!isalnum(s[p]) && s[p]!='_' && s[p]!='.' && s[p]!=':') {
            state = -1; continue;
          }
          break;
        case -3: // string
          if (n == 0 && s[p] == q) {
            state = -1;
          } else if (n == 0 && s[p]=='\\') {
            p += 1;
          } else if (n && s[p]==']' && p>=n && s[p-n]==']') {
            const char *t = s + p - n + 1;
            while (*t == '=')
              t += 1;
            if (t == s + p)
              state = -1;
          }
          break;
        case -4: // comment
          if (n == 0 && (s[p] == '\n' || s[p] == '\r')) {
            state = -1;
          } else if (n && s[p]==']' && p>=n && s[p-n]==']') {
            const char *t = s + p - n + 1;
            while (*t == '=')
              t += 1;
            if (t == s + p)
              state = -1;
          }
          break;
        case -5: // opening function
          if (s[p] != ' ') {
            state = -1;
            p--;
          }
          break;
        }
      p += 1;
    }
  return state;
}

static bool rtty_inited = false;
static QByteArray rtty_history;
static QList<QByteArray> *rtty_completions;

static char *
rtty_compentry(const char *text, int state)
{
  if (rtty_completions->isEmpty()) return 0;
  QByteArray b = text + rtty_completions->takeFirst();
  return strdup(b.constData());
}

static char *
rtty_dummycomp(const char *text, int state)
{
  static int st = 0;
  if (state == 0) {
    st = 0;
  }
  if (st == 2) {
    return 0;
  } else if (st == 1) {
    st = 2;
    return strdup("/");
  } else if (st == 0) {
    st = 1;
    return strdup("\\");
  }
  return 0;
}

static char **
rtty_complete(const char *text, int start, int end)
{
  int state = rtty_lex(rl_line_buffer, end, start);
  // default filename completion in string
  if (state == -3) 
    return 0;
  // no default completion
  rl_attempted_completion_over = 1;
  // help completion if opening function
  if (state == -5) {
    // copy keyword
    int len = end - start - 1;
    char *keyword = (char *)malloc(sizeof(char)*(len+1));
    int i=0;
    for (; i<len; i++) {
      if (rl_line_buffer[start+i] == '(') break;
      keyword[i] = rl_line_buffer[start+i];
    }
    keyword[i] = 0;
    // prev keyword
    static char *prevkw = NULL;
    if (prevkw == NULL || strcmp(prevkw,keyword) != 0) {
      if (prevkw != NULL) free(prevkw);
      prevkw = strdup(keyword);
      return 0;
    } else {
      if (prevkw != NULL) free(prevkw);
      prevkw = strdup(keyword);
    }
    // get help for keyword
    if (console && console->lua) {
      QtLuaLocker lua(console->lua, 250);
      struct lua_State *L = lua;
      if (lua) {
        lua_getfield(L, LUA_GLOBALSINDEX, "help");
        if (lua_gettop(L)) {
          printf("\n");
          lua_pushstring(L, keyword);
          lua_pcall(L, 1, 0, 0);
        }
      }
    }
    // done
    char **result = rl_completion_matches(text, rtty_dummycomp);
    return result;
  }
  // no completion unless in identifier
  if (state != -2 || start >= end) 
    return 0;
  // complete 
  const char *stem = rl_line_buffer + start;
  int stemlen = end - start;
  QList<QByteArray> completions;
  completions.clear();
  // complete: keywords
  for (const char **k = rtty_keywords; *k; k++)
    if (!strncmp(stem, *k, stemlen))
      completions += QByteArray(*k + stemlen);
  // complete: identifiers
  if (console && console->lua)
    {
      QtLuaLocker lua(console->lua, 250);
      struct lua_State *L = lua;
      if (lua)
        {
          lua_pushcfunction(L, luaQ_complete);
          lua_pushlstring(L, stem, stemlen);
          if (!lua_pcall(L, 1, 1, 0) && lua_istable(L, -1)) {
            int n = lua_objlen(L, -1);
            for (int j=1; j<=n; j++) {
              lua_rawgeti(L, -1, j);
              completions += QByteArray(lua_tostring(L, -1));
              lua_pop(L, 1);
            }
          }
          lua_pop(L, 1);
        }
    }
  qSort(completions.begin(), completions.end());
  if (completions.isEmpty())
    return 0;
  rtty_completions = &completions;
  char **result = rl_completion_matches(text, rtty_compentry);
  rtty_completions = 0;
#if RL_READLINE_VERSION >= 0x0600
  rl_completion_suppress_append = 1;
#endif
  return result;
}

static void
rtty_prep()
{
  // readline
  static QByteArray progname;
  QFileInfo fileinfo = QCoreApplication::applicationFilePath();
  progname = fileinfo.baseName().toLocal8Bit();
  rl_readline_name = progname.data();
  rl_getc_function = rtty_getchar;
  rl_attempted_completion_function = rtty_complete;
  rl_completer_quote_characters = "\"'";
#if RL_READLINE_VERSION < 0x0600
  rl_completion_append_character = '\0';
#endif
#if RL_READLINE_VERSION > 0x402
  rl_set_paren_blink_timeout(250000);
  rl_bind_key (')', rl_insert_close);
  rl_bind_key (']', rl_insert_close);
  rl_bind_key ('}', rl_insert_close);
  rl_variable_bind("comment-begin","-- ");
#endif
  rl_initialize();
  // history
  rtty_history = ".luahistory";
  const char *home = getenv("HOME");
  const char *luaHistory = getenv("LUA_HISTORY");
  const char *luaHistSize = getenv("LUA_HISTSIZE");
  int histSize = 1000;
  if (luaHistory && luaHistory[0])
    rtty_history = luaHistory;
  else if (home && home[0])
    rtty_history = QByteArray(home) + "/" + rtty_history;
  if (luaHistSize && luaHistSize[0])
    histSize = strtol(luaHistSize, 0, 10);
  using_history();
  stifle_history(qBound(25,histSize,250000));
  read_history(rtty_history.constData());
  // done
  rtty_inited = true;
}


static QByteArray
rtty_readline(const char *prompt)
{
  // prep
  if (! rtty_inited) 
    rtty_prep();
  // flush
  FILE *ferr = ((console) ? console->trueStderr : stderr);
  FILE *fout = ((console) ? console->trueStdout : stdout);
  fflush(ferr);
  fflush(fout);
  // readline
  rl_instream = stdin;
  rl_outstream = fout;
  setbuf(stdin, 0);
  clearerr(stdin);
  rtty_status = RttyOk;
  char *s = readline(prompt);
  // cleanup and return
  QByteArray ba;
  if (s && rtty_status == RttyOk)
    ba = s;
  if (!s && rtty_status == RttyOk)
    rtty_status = RttyEof;
  if (s)
    free(s);
  if (rtty_status != RttyOk)
    fputs("\n", fout);
  fflush(fout);
  return ba;
}


static void
rtty_done()
{
  if (rtty_inited)
    write_history(rtty_history.constData());
}


#else //!HAVE READLINE


QByteArray
rtty_readline(const char *prompt)
{
  // flush
  FILE *ferr = ((console) ? console->trueStderr : stderr);
  FILE *fout = ((console) ? console->trueStdout : stdout);
  fflush(ferr);
  fprintf(fout,"%s",prompt);
  fflush(fout);
  // read
  QByteArray ba;
  setbuf(stdin, 0);
  clearerr(stdin);
  rtty_status = RttyOk;
  for(;;)
    {
      int c = rtty_getchar(stdin);
      if (c == EOF || c == '\r' || c == '\n' || rtty_status != RttyOk)
        {
          if (c != '\n')
            fputs("\n", fout);
          if (c == EOF && ! ba.size() && rtty_status == RttyOk)
            rtty_status = RttyEof;
          break;
        }
      ba += c;
    }
  fflush(fout);
  return ba;
}

#endif //!HAVE READLINE




// ----------------------------------------
// Various callbacks


static void 
message_handler(QtMsgType type, const char *msg)
{
  FILE *ferr = stderr;
  if (console)
    ferr = console->trueStderr;
  switch (type) 
    {
    case QtDebugMsg:
      fprintf(ferr, "# Debug: %s\n", msg);
      break;
    case QtWarningMsg:
      fprintf(ferr, "# Warning: %s\n", msg);
      break;
    case QtCriticalMsg:
      fprintf(ferr, "# Critical: %s\n", msg);
      break;
    case QtFatalMsg:
      fprintf(ferr, "# Fatal: %s\n", msg);
      abort();
    }
}


static void
handle_sigint(int)
{
  if (console)
    console->sigint();
}



// ----------------------------------------
// QLuaConsole::Private


QLuaConsole::Private::Private(QLuaConsole *parent)
  : QThread(parent), 
    q(parent),
    throttleCount(0),
    throttleActive(false),
    killConsole(false),
    captureOutput(false),
    printCapturedOutput(true),
    redirected(false),
    trueStdout(0),
    trueStderr(0),
    breakStopsLua(false),
    ctrlCHandler(0)
{
  // save unique pointer
  Q_ASSERT(!console);
  console = this;

  // hold all signals from the main thread.
  // signal processing happens in 
  // the console thread (see run)
  hold_signals(&savedSigSet);    // <-- save the sigset before
  
  // create pipes
  if (pipe(stdoutPipe) < 0)
    qFatal("qlua console: unable to create output pipe");    
  if (pipe(commandPipe) < 0) 
    qFatal("qlua console: unable to create command pipe");    

  // duplicate stdout/stderr
  int stdoutFd = dup(fileno(stdout));
  int stderrFd = dup(fileno(stderr));
  if (stdoutFd < 0 || stderrFd < 0)
    qFatal("qlua console: cannot duplicate stdout/stderr descriptors");
  trueStdout = fdopen(stdoutFd, "w");
  trueStderr = fdopen(stderrFd, "w");
  if (!trueStdout || !trueStderr)
    qFatal("qlua console: cannot open duplicate stdout/stderr");

  // connect consoleOutput
  connect(this, SIGNAL(sigConsoleOutput(QByteArray)),
          this, SLOT(slotConsoleOutput(QByteArray)) );
}


QLuaConsole::Private::~Private()
{
  Q_ASSERT(console == this);
  // console thread
  command(KillCmd);
  wait();
  // file descriptors
  redirect(false);
  if (trueStdout)
    fclose(trueStdout);
  if (trueStderr)
    fclose(trueStderr);
  ::close(stdoutPipe[0]);
  ::close(stdoutPipe[1]);
  ::close(commandPipe[0]);
  ::close(commandPipe[1]);
  // signals
  release_signals(&savedSigSet);
  set_sigint_handler(SIG_DFL);
  // done
#if HAVE_READLINE
  rtty_done();
#endif
  console = 0;
}


void
QLuaConsole::Private::command(Command c)
{
  char command = (char)c;
  (void) ::write(commandPipe[1], &command, 1);
}


void
QLuaConsole::Private::sigint()
{
  // Under windows, this runs in a separate thread.
  // Under unix, this should run from the console thread
  // because we take great pains to use pthread_sigmask().
  // Alas the implementation of pthread signal handling is often poor
  // and readline changes our signal masks with sigprocmask()
  // with unpredictable effect on some machines.
  // So we send a command to the console thread which
  // will do the signaling and hope that readline
  // will not subsequently mess the console beyond repair.
  command(BreakCmd);
}


void
QLuaConsole::Private::sethandler()
{
  if (ctrlCHandler == ctrlCDefault)
    set_sigint_handler(SIG_DFL);
  else if (ctrlCHandler == ctrlCIgnore)
    set_sigint_handler(SIG_IGN);    
  else if (ctrlCHandler == ctrlCBreak)
    set_sigint_handler(handle_sigint);
  else
    set_sigint_handler(ctrlCHandler);
}


void 
QLuaConsole::Private::redirect(bool flag)
{
  QMutexLocker lock(&mutex);
  if (flag != redirected)
    {
      command(NoCmd);
      fflush(stdout);
      fflush(stderr);
      fflush(trueStdout);
      fflush(trueStderr);
      if (flag)
        {
          dup2(stdoutPipe[1], fileno(stdout));
          dup2(stdoutPipe[1], fileno(stderr));
        }
      else
        {
          dup2(fileno(trueStdout), fileno(stdout));
          dup2(fileno(trueStderr), fileno(stderr));
        }
      redirected = flag;
    }
  setbuf(stdout, 0);
  setbuf(stderr, 0);
}


void
QLuaConsole::Private::run()
{
  // --- This runs from the console thread.
  // accept signals in this thread.
  set_sigint_handler(SIG_DFL);
  release_signals(&savedSigSet);
  // loop
  while (! killConsole)
    {
      int fds[2];
      int nfd = 0;
      if (! throttleActive)
        fds[nfd++] = stdoutPipe[0];
      fds[nfd++] = commandPipe[0];
      int fd = wait_for_input(nfd, fds);
      mutex.lock();
      if (! throttleActive)
        copyout();
      if (fd == commandPipe[0])
        {
          char c = (char)NoCmd;
          if (::read(commandPipe[0], &c, 1) > 0)
            switch( (enum Command)c )
              {
              case HandlerCmd:
                sethandler();
                break;
              case DrainCmd:
                msleep(10);
                copyout(throttleActive = false);
                drain.wakeAll();
                break;
              case BreakCmd:
                if (lua && breakStopsLua) 
                  lua->stop(true);
                emit q->ttyBreak();
                break;
              case ReadlineCmd:
                readline();
                break;
              case KillCmd:
                killConsole = true;
              default:
                break;
              }
        }
      pulse.wakeAll();
      mutex.unlock();
    }
  // reset signals
  set_sigint_handler(SIG_DFL);
}


int
QLuaConsole::Private::getchar()
{
  // --- This runs from the console thread,
  //     mutex is unlocked.

  // is data available already?
  int fd = fileno(stdin);
  if (wait_for_input(1, &fd, false) >= 0)
    return fgetc(stdin);
  // we must wait
  QMutexLocker lock(&mutex);
  while (! killConsole)
    {
      int fds[3];
      int nfd = 0;
      if (! throttleActive)
        fds[nfd++] = stdoutPipe[0];
      fds[nfd++] = commandPipe[0];
      fds[nfd++] = fileno(stdin);
      lock.unlock();
      int fd = wait_for_input(nfd, fds);
      lock.relock();
      if (! throttleActive)
        copyout();
      if (fd == commandPipe[0])
        {
          char c = (char)NoCmd;
          if (::read(commandPipe[0], &c, 1) > 0)
            switch( (enum Command)c )
              {
              case HandlerCmd:
                sethandler();
                break;
              case DrainCmd:
                msleep(10);
                copyout(throttleActive = false);
                drain.wakeAll();
                break;
              case BreakCmd:
                if (lua && breakStopsLua) 
                  lua->stop(true);
                emit q->ttyBreak();
                break;
              case KillCmd:
                killConsole = true;
              case AbortCmd:
                rtty_status = RttyAbort;
              default:
                break;
              }
        }
      if (rtty_status == RttyAbort)
        break;
      if (fd == fileno(stdin))
        return fgetc(stdin);
    }
  return EOF;
}

           
void
QLuaConsole::Private::copyout(bool throttle)
{
  // --- This runs from the console thread,
  //     mutex is locked.
  char buffer[1024];
  int fd = stdoutPipe[0];
  while (!throttleActive && wait_for_input(1, &fd, false) >= 0)
    {
      int sz = ::read(fd, buffer, sizeof(buffer));
      if (sz < 0)
        return;
      QByteArray ba(buffer, sz);
      emit sigConsoleOutput(ba);
      if (++throttleCount > 16)
        throttleActive = throttle;
      if (printCapturedOutput)
        fwrite(buffer, 1, sz, trueStdout);
    }
}


void 
QLuaConsole::Private::slotConsoleOutput(QByteArray ba)
{
  QByteArray bb;
  bb.reserve(ba.size());
  const char *s = ba.constData();
  const char *e = s + ba.size();
  while (s < e)
    {
      char c = *s++;
      if (!isascii(c) || isspace(c) || isprint(c))
        bb += c;
    }
  emit q->consoleOutput(bb);
  QMutexLocker locker(&mutex);
  if (! --throttleCount && throttleActive)
    QTimer::singleShot(1, this, SLOT(slotCommandNoCmd()));
}


void 
QLuaConsole::Private::slotCommandNoCmd()
{
  QMutexLocker locker(&mutex);
  throttleActive = false;
  command(NoCmd);
}


void
QLuaConsole::Private::readline()
{
  // --- This runs from the console thread,
  //     mutex is locked.
  QByteArray prompt = this->prompt;
  QByteArray ba;
  mutex.unlock();
  ba = rtty_readline(prompt.constData()); 
  mutex.lock();
  switch (rtty_status)
    {
    case RttyOk:
      emit q->ttyInput(ba);
      break;
    case RttyEof:
      emit q->ttyEndOfFile();
    case RttyAbort:
      // drain output
      clearerr(stdin);
      int fd = fileno(stdin);
      if (wait_for_input(1, &fd, false) >= 0)
        fgetc(stdin);
      break;
    }
}



// ----------------------------------------
// QLuaConsole


/*! \class QLuaConsole
  This object handles the console input and output.
  When \a captureOutput is sets, it captures everything
  printed on the standard output and broadcast it 
  using signal \a consoleOutput.
  Calling \a readLine causes this object to 
  asynchronously read one line of command on the terminal.
  The object will indicate that the line is available
  using signal \a ttyInput. */

/* Constructor */

QLuaConsole::QLuaConsole(QObject *parent)
  : QObject(parent), d(new Private(this))
{
  qInstallMsgHandler(message_handler);
  d->start();
}


/*! \property QLuaConsole::captureOutput
  Set this to capture the standard output and
  broadcast it by emiting signals \a consoleOutput.
 */

bool 
QLuaConsole::captureOutput() const
{
  return d->captureOutput;
}

void 
QLuaConsole::setCaptureOutput(bool flag)
{
  d->captureOutput = flag;
  d->redirect(flag);
  if (d->isRunning())
    d->command(NoCmd); 
}


/*! \property QLuaConsole::printCapturedOutput
  Set this to echo all captured output strings on the 
  true standard output. */

bool 
QLuaConsole::printCapturedOutput() const
{
  return d->printCapturedOutput;
}

void 
QLuaConsole::setPrintCapturedOutput(bool flag)
{
  d->printCapturedOutput = flag;
}


typedef QLuaConsole::CtrlCHandler CtrlCHandler;

const CtrlCHandler QLuaConsole::ctrlCDefault = (CtrlCHandler)(-1);
const CtrlCHandler QLuaConsole::ctrlCIgnore = (CtrlCHandler)(-2);
const CtrlCHandler QLuaConsole::ctrlCBreak = (CtrlCHandler)(-3);

/*! Sets the Ctrl-C handler for the process.
  Special values \a QLuaConsole::ctrlCDefault 
  and \a QLuaConsole::ctrlCIgnore are equivalent
  to the usual signal handlers \a SIG_DFL and \a SIG_IGN.
  Special value \a QLuaConsole::ctrlCBreak runs
  a handler that emits signal \a ttyBreak(). */

CtrlCHandler 
QLuaConsole::setCtrlCHandler(CtrlCHandler handler)
{
  CtrlCHandler old = d->ctrlCHandler;
  if (handler)
    {
      d->ctrlCHandler = handler;
      d->command(HandlerCmd);
    }
  return (old) ? old : ctrlCDefault;
}



/*! Sets the lua interpreter that will be used by subsequent 
  readline operations for completion and locking. 
  When flags \a breakStopsLua, pressing CTRL-C stops
  the engine using \a QtLuaEngine::stop().
  This is useful when running in single thread mode
  because the main thread would never get a chance
  to receive the \a ttyBreak signal since it is busy
  running Lua. */

void 
QLuaConsole::setQtLuaEngine(QtLuaEngine *lua, bool breakStopsLua)
{
  d->lua = lua;
  d->breakStopsLua = breakStopsLua;
}


/*! Start reading a line from the console with the provided prompt.
  This is ignored if the console is already reading a line.
  Cause the emission of signal \a ttyLineRead
  when the line has been read. */

void 
QLuaConsole::readLine(QByteArray prompt)
{
  QMutexLocker lock(&d->mutex);
  d->prompt = prompt;
  d->command(ReadlineCmd);
}


/*! Abort the current \a readLine operation.
  This is ignored if the console is not reading a line. */

void 
QLuaConsole::abortReadLine()
{
  QMutexLocker lock(&d->mutex);
  d->command(AbortCmd);
  d->pulse.wait(&d->mutex);
}


/*! Causes the current \a readLine operation
  to redisplay the line because it may have been garbled by extra printouts. 
  This does not work yet. */

void 
QLuaConsole::redisplayReadLine()
{
  d->command(RedisplayCmd);
}


/*! Save line into the readline history when available */

void
QLuaConsole::addToHistory(QByteArray line)
{
#if HAVE_READLINE
  if (line.size() > 0)
    add_history(line.constData());
#else
  (void)line;
#endif
}



/*! Make sure that all output has been reemitted,
  provided that one is not reading a line on the terminal. */

void
QLuaConsole::drainOutput()
{
  if (d->captureOutput)
    {
      fflush(stdout);
      fflush(stderr);
      QMutexLocker lock(&d->mutex);
      d->command(DrainCmd);
      d->drain.wait(&d->mutex);
      lock.unlock();
      QCoreApplication::sendPostedEvents();
    }
}



/*! \signal QLuaConsole::ttyBreak()
  Indicate that we have received the unix signal \a SIGINT
  which usually is the effect of typing Ctrl+C in the console. */

/*! \signal QLuaConsole::consoleOutput(QByteArray output)
  Indicate that \a output has been recently
  printed on the standard output. Property \a captureOutput
  must be set for this to work. */

/*! \signal QLuaConsole::ttyInput(QByteArray input)
  Indicate that a line had been read on the terminal.
  This is a consequence of a prior call to \a readLine(). */

/*! \signal QLuaConsole::ttyEndOfFile()
  Indicate that we have reached the end-of-file on the terminal.
  The current \a readLine operation is aborted. */


// ----------------------------------------
// MOC

#include "qluaconsole_unix.moc"





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ( "\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" )
   End:
   ------------------------------------------------------------- */

