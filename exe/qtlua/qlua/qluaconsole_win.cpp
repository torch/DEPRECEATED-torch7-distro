// -*- C++ -*-


#include "qluaconf.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <exception>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <io.h>

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

#include "windows.h"



// ----------------------------------------
// QLuaConsole StdoutThread


namespace {

  class StdoutThread : public QThread
  {
    Q_OBJECT
  public:
    ~StdoutThread();
    StdoutThread(QLuaConsole::Private *d);
    void run();
  signals:
    void consoleOutput(QByteArray output);
  public:
    QLuaConsole::Private *d;
    FILE * volatile fout;
    const char *sync;
    int fds[2];
    QMutex mutex;
    QWaitCondition sCond;
    QWaitCondition tCond;
    int tCount;
    bool tActive;
  private:
  };
  
  StdoutThread::~StdoutThread()
  {
    terminate();
    wait();
  }

  StdoutThread::StdoutThread(QLuaConsole::Private *d)
    : d(d), 
      fout(0), 
      sync("\027SYNC\007"),
      tCount(0),
      tActive(false)
  {
    fds[0] = fds[1] = -1;
    if (_pipe(fds,1024,_O_BINARY|_O_NOINHERIT) < 0)
      qFatal("Cannot create stdout pipe");
  }
  
  void
  StdoutThread::run()
  {
    int spos = 0;
    QMutexLocker locker(&mutex);
    for(;;)
      {
        const int bufferSize = 1024;
        char ebuffer[bufferSize+8];
        char *buffer = ebuffer + 8;
        locker.unlock();
        setTerminationEnabled(true);
        int bufsiz = _read(fds[0], buffer, bufferSize);
        setTerminationEnabled(false);
        locker.relock();
        if (bufsiz <= 0)
          break;
        char *s = ebuffer;
        for (int i=0; i<bufsiz; i++)
          {
            // sync
            if (buffer[i] == sync[spos])
              { 
                if (! sync[++spos])
                  {
                    sCond.wakeOne();
                    spos = 0;
                  }
                continue;
              }
            if (spos > 0)
              {
                memcpy(s, sync, spos);
                s += spos;
                spos = 0;
              }
            *s++ = buffer[i];
          }
        bufsiz = s - ebuffer;
        if (bufsiz <= 0)
          continue;
        QByteArray ba(ebuffer, bufsiz);
        FILE *f = fout;
        if (f) 
          fprintf(f, "%s", ba.constData());
        emit consoleOutput(ba);
        if (++tCount > 16)
          {
            tActive = true;
            tCond.wait(&mutex);
          }
      }
  }

}



// ----------------------------------------
// QLuaConsole ReadlineThread


namespace {

  class ReadlineThread : public QThread
  {
    Q_OBJECT
  public:
    ReadlineThread(QLuaConsole::Private *p);
    ~ReadlineThread();
    void run();
    void abort();
    void setPrompt(QByteArray p) { prompt = p; }
    void setEngine(QtLuaEngine *l) { lua = l; }
    void setStdout(FILE *fo) { fout=fo; }
  signals:
    void ttyInput(QByteArray input);
    void ttyBreakOrEof();
  private:
    QLuaConsole::Private *d;
    QPointer<QtLuaEngine> lua;
    FILE *fout;
    QByteArray prompt;
    HANDLE tid;
    bool cancelled;
  };

  ReadlineThread::~ReadlineThread()
  {
    terminate();
    wait();
  }

  ReadlineThread::ReadlineThread(QLuaConsole::Private *d)
    : d(d), fout(stdout), tid(0), cancelled(false)
  {
    prompt = "> ";
  }
  
  void
  ReadlineThread::run()
  {
    // This is where we could use a readline replacement.
    size_t bufsiz = 0;
    const char *bufdat = 0;
    if (fout)
      {
        fflush(fout);
        fputs(prompt.constData(), fout);
        fflush(fout);
      }
    char buffer[1024];
    cancelled = false;
    setTerminationEnabled(true);
    bufdat = fgets(buffer, sizeof(buffer)-1, stdin);
    bool was_cancelled = cancelled;
    cancelled = true;
    setTerminationEnabled(false);
    if (bufdat && bufdat[0] == 4)
      bufdat = 0;
    bufsiz = (bufdat) ? strlen(bufdat) : 0;
    if (was_cancelled)
      return;
    if (bufsiz<=0 || bufdat[bufsiz-1] != '\n')
      fputs("\n", fout);
    if (bufdat && bufsiz>=0)
      emit ttyInput(QByteArray(bufdat, bufsiz));
    else {
      // Ctrl-c interrupts fgets() but makes it 
      // indistinguishable from an end of file.
      // Therefore we emit ttyBreakOrEof() and the connected slot
      // in class Private will decide whether this is a true EOF.
      emit ttyBreakOrEof();
      // Linger a little bit to make sure ctrl-c arrives.
      msleep(250);
    }
  }


  void
  ReadlineThread::abort()
  {
    if (isRunning())
      {
        if (! cancelled)
          {
            cancelled = true;
            // send a fake enter to terminate line processing!
            INPUT_RECORD rec[2];
            rec[0].EventType = KEY_EVENT;
            rec[0].Event.KeyEvent.bKeyDown = TRUE;
            rec[0].Event.KeyEvent.dwControlKeyState = 0;
            rec[0].Event.KeyEvent.uChar.AsciiChar = '\r';
            rec[0].Event.KeyEvent.wRepeatCount = 1;
            rec[0].Event.KeyEvent.wVirtualKeyCode = VK_RETURN;
            rec[0].Event.KeyEvent.wVirtualScanCode = 43;
            rec[1].EventType = KEY_EVENT;
            rec[1].Event.KeyEvent.bKeyDown = FALSE;
            rec[1].Event.KeyEvent.dwControlKeyState = 0;
            rec[1].Event.KeyEvent.uChar.AsciiChar = '\r';
            rec[1].Event.KeyEvent.wRepeatCount = 1;
            rec[1].Event.KeyEvent.wVirtualKeyCode = VK_RETURN;
            rec[1].Event.KeyEvent.wVirtualScanCode = 43;
            HANDLE chnd = GetStdHandle(STD_INPUT_HANDLE);
            DWORD written = 0;
            WriteConsoleInput(chnd, rec, 2, &written);
          }
        // wait for termination of the readline thread.
        wait();
       }
  }
  
}




// ----------------------------------------
// QLuaConsole::Private


struct QLuaConsole::Private : public QObject
{
  Q_OBJECT
public:
  QLuaConsole *q;
  bool captureOutput;
  bool printCapturedOutput;
  bool redirected;
  bool volatile interrupted;
  FILE *trueStdout;
  FILE *trueStderr;
  StdoutThread sThread;
  ReadlineThread rThread;
  QPointer<QtLuaEngine> lua;
  bool breakStopsLua;                         
public slots:
  void consoleXon();
  void consoleOutput(QByteArray ba);
  void ttyInput(QByteArray ba);
  void ttyBreakOrEof();
public:
  ~Private();
  Private(QLuaConsole *parent);
  void sigint();
  void redirect(bool flag);
};


static QLuaConsole::Private *console = 0;


static void 
message_handler(QtMsgType type, const char *msg)
{
  FILE *ferr = stderr;
  if (console)
    ferr = console->trueStderr;
  switch (type) 
    {
    case QtDebugMsg:
	  if (ferr)
        fprintf(ferr, "# Debug: %s\n", msg);
      break;
    case QtWarningMsg:
	  if (ferr)
        fprintf(ferr, "# Warning: %s\n", msg);
      break;
    case QtCriticalMsg:
	  if (ferr)
        fprintf(ferr, "# Critical: %s\n", msg);
      break;
    case QtFatalMsg:
	  if (ferr)
        fprintf(ferr, "# Fatal: %s\n", msg);
      abort();
    }
}


static void
handle_sigint(int)
{
  signal(SIGINT, handle_sigint);
  if (console)
    console->sigint();
}


QLuaConsole::Private::Private(QLuaConsole *parent)
  : QObject(parent), 
    q(parent),
    captureOutput(false),
    printCapturedOutput(true),
    redirected(false),
    interrupted(false),
    trueStdout(0),
    trueStderr(0),
    sThread(this),
    rThread(this),
    breakStopsLua(false)
{
  // unique pointer
  Q_ASSERT(!console);
  console = this;
  // message handler
  qInstallMsgHandler(message_handler);
  // duplicate stdout/stderr
  int stdoutFd = _dup(_fileno(stdout));
  if (stdoutFd >= 0)
    trueStdout = _fdopen(stdoutFd, "wt");
  int stderrFd = _dup(_fileno(stderr));
  if (stderrFd >= 0)
    trueStderr = _fdopen(stderrFd, "wt");
  // connections
  connect(&sThread, SIGNAL(consoleOutput(QByteArray)),
          this, SLOT(consoleOutput(QByteArray)) );
  connect(&rThread, SIGNAL(ttyInput(QByteArray)),
          this, SLOT(ttyInput(QByteArray)) );
  connect(&rThread, SIGNAL(ttyBreakOrEof()),
          this, SLOT(ttyBreakOrEof()) );
  // signals
  signal(SIGINT, SIG_DFL);
}


QLuaConsole::Private::~Private()
{
  Q_ASSERT(console == this);
  // file descriptors
  redirect(false);
  if (trueStdout)
    fclose(trueStdout);
  if (trueStderr)
    fclose(trueStderr);
  // signals
  signal(SIGINT, SIG_DFL);
  // done
  console = 0;
}


void
QLuaConsole::Private::sigint()
{
  if (rThread.isRunning())
    interrupted = true;
  else
    {
      if (lua && breakStopsLua) 
        lua->stop(true);
      emit q->ttyBreak();
    }
}


void 
QLuaConsole::Private::consoleOutput(QByteArray ba)
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
  QMutexLocker locker(&sThread.mutex);
  if (! --sThread.tCount && sThread.tActive)
    QTimer::singleShot(1, this, SLOT(consoleXon()));
}


void 
QLuaConsole::Private::consoleXon()
{
  QMutexLocker locker(&sThread.mutex);
  sThread.tActive = false;
  sThread.tCond.wakeOne();
}


void 
QLuaConsole::Private::ttyInput(QByteArray ba)
{
  rThread.wait();
  emit q->ttyInput(ba);
}


void 
QLuaConsole::Private::ttyBreakOrEof()
{
  rThread.wait();
  if (interrupted)
    emit q->ttyBreak();
  else
    emit q->ttyEndOfFile();
}


void 
QLuaConsole::Private::redirect(bool flag)
{
  if (flag != redirected)
    {
      fflush(stdout);
      fflush(stderr);
      if (trueStdout)
        fflush(trueStdout);
      if (trueStderr)
        fflush(trueStderr);
      int fd1 = sThread.fds[1];
      int fd2 = fd1;
      if (! trueStdout) 
        stdout->_file = 1;
      else if (! flag)
        fd1 = _fileno(trueStdout);
      if (! trueStderr) 
        stderr->_file = 2;
      else if (! flag)
        fd2 = _fileno(trueStderr);
      _dup2(fd1, _fileno(stdout));
      _dup2(fd2, _fileno(stderr));
      redirected = flag;
      if (redirected && ! sThread.isRunning())
        sThread.start();
    }
  setbuf(stdout, NULL);
  setbuf(stderr, NULL);
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
  if (flag && ! d->rThread.isRunning())
    d->sThread.start();
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
  d->sThread.fout = (flag) ? d->trueStdout : NULL;
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
  if (handler == ctrlCDefault)
    return signal(SIGINT, SIG_DFL);
  else if (handler == ctrlCIgnore)
    return signal(SIGINT, SIG_IGN);    
  else if (handler == ctrlCBreak)
    return signal(SIGINT, handle_sigint);
  else
    return signal(SIGINT, handler);
}


/*! Sets the lua interpreter that will be used by subsequent 
  readline operations for completion and locking. */

void 
QLuaConsole::setQtLuaEngine(QtLuaEngine *lua, bool breakStopsLua)
{
  d->lua = lua;
  d->breakStopsLua = breakStopsLua;
  d->rThread.setEngine(lua);
}


/*! Start reading a line from the console with the provided prompt.
  This is ignored if the console is already reading a line.
  Cause the emission of signal \a ttyLineRead
  when the line has been read. */

void 
QLuaConsole::readLine(QByteArray prompt)
{
  if (! d->rThread.isRunning())
    {
      d->rThread.setPrompt(prompt);
      d->rThread.setStdout(d->trueStdout);
      d->rThread.start();
      d->interrupted = false;
    }
}


/*! Abort the current \a readLine operation.
  This is ignored if the console is not reading a line. */

void 
QLuaConsole::abortReadLine()
{
  if (d->rThread.isRunning())
    d->rThread.abort();
}


/*! Causes the current \a readLine operation
  to redisplay the line because it may have been garbled by extra printouts. 
  This does not work yet. */

void 
QLuaConsole::redisplayReadLine()
{
}


/*! Save line into the readline history when available */

void
QLuaConsole::addToHistory(QByteArray line)
{
  (void)line;
}


/*! Make sure that all output has been reemitted */

void
QLuaConsole::drainOutput()
{
  if (d->captureOutput)
    {
      QMutexLocker locker(&d->sThread.mutex);
      fprintf(stdout, "%s", d->sThread.sync);
      fflush(stdout);
      fflush(stderr);
      d->sThread.tCount -= 10000;
      d->sThread.tActive = false;
      d->sThread.tCond.wakeAll();
      d->sThread.sCond.wait(&d->sThread.mutex);
      d->sThread.tCount += 10000;
      locker.unlock();
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

#include "qluaconsole_win.moc"



/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ( "\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" )
   End:
   ------------------------------------------------------------- */

