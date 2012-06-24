// -*- C++ -*-

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <locale.h>

#include <QtGlobal>
#include <QDebug>
#include <QFont>
#include <QVector>

#include "qluaapplication.h"

#ifdef Q_WS_X11
extern "C" Display *XOpenDisplay(char*);
extern "C" int XCloseDisplay(Display*);
# ifdef HAVE_XINITTHREADS
extern "C" int XInitThreads();
# endif
#endif


static bool
optionTakesArgument(const char *s)
{
  // keep this in sync as much as possible
  static const char *optionsWithArg[] =
    { // defined by Qt4.
      "-style", "-session", "-stylesheet",
#ifdef Q_WS_X11
      "-display", "-font", "-fn", "-button", "-btn",
      "-background", "-bg", "-foreground", "-fg",
      "-name", "-title", "-geometry", "-im", 
      "-ncols", "-visual", "-inputstyle", 
#endif
      // defined by QLuaApplication
      "-l", "-e",
      0 };
  for (int i=0; optionsWithArg[i]; i++)
    if (!strcmp(s, optionsWithArg[i]))
      return true;
  return false;
}


static bool
optionIsIde(const char *s, QByteArray &ideStyle)
{
  if (strncmp(s, "-ide", 4))
    return false;
  if (s[4] == '=')
    ideStyle = s+5;
  else if (s[4])
    return false;
  return true;
}


int
main(int argc, char **argv)
{ 
  int argn = 0;
  bool ide = false;
  QByteArray ideStyle;
  bool graphics = true;
  bool onethread = false;
  QVector<char*> args_for_both;
  char *displayName = 0;
  // Locale
#ifdef LC_ALL
  ::setlocale(LC_ALL, "");
  ::setlocale(LC_NUMERIC, "C");
#endif
  // Split argument list
  while (argn<argc)
    {
      char *s = argv[argn];
      if (argn>0 && (s[0]!='-' || !strcmp(s,"-") || !strcmp(s,"--")))
        break;
      else if (!strcmp(s, "-nographics"))
        graphics = false;
      else if (!strcmp(s, "-onethread"))
        onethread = true;
      else if (optionIsIde(s, ideStyle))
        ide = true;
      else if (!strcmp(s, "-h") && argn <= 1)
        graphics = ide = false;
      if (!strcmp(s, "-display") && argn+1 < argc)
        displayName = argv[argn+1];
      if (optionTakesArgument(s) && argn+1 < argc)
        args_for_both += argv[argn++];
      args_for_both += argv[argn++];
    }
  args_for_both += 0;
  int argc_for_both = args_for_both.size() - 1;
  char **argv_for_both = args_for_both.data();

  // Special behavior for X11.
#ifdef Q_WS_X11
# if defined(HAVE_XINITTHREADS) && (QT_VERSION < 0x40400)
  XInitThreads();
# endif
  Display *display = XOpenDisplay(displayName);
  if (display)
    XCloseDisplay(display);
  else if (graphics)
    qWarning("Unable to connect X11 server (continuing with -nographics)");
  graphics &= !!display;
#endif
  
  // Create application object
  QLuaApplication app(argc_for_both, argv_for_both, graphics, onethread);

  // Construct arguments for lua
  int i = 0;
  QVector<char*> args_for_lua;
  if (i < argc_for_both)
    args_for_lua += argv_for_both[i++];
  while(ide && i<argc_for_both && !optionIsIde(argv_for_both[i], ideStyle))
    args_for_lua += argv_for_both[i++];
#if defined(WINQLUA) || defined(MACQLUA)
  if (graphics && argc_for_both <= 1 && argn >= argc)
    ide = true;
#endif
  if (ide)
    {
      if (ideStyle.isEmpty())
        ideStyle = "-e qtide.start()";        
      else
        ideStyle = "-e qtide.start('" + ideStyle + "')";
      args_for_lua += const_cast<char*>("-lqtide");
      args_for_lua += const_cast<char*>(ideStyle.constData());
    }
  while(i < argc_for_both)
    args_for_lua += argv_for_both[i++];
  while(argn < argc)
    args_for_lua += argv[argn++];
  args_for_lua += 0;
  
  // Call main function
  int argc_for_lua = args_for_lua.size() - 1;
  char **argv_for_lua = args_for_lua.data();
  return app.main(argc_for_lua, argv_for_lua);
}





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ( "\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" )
   End:
   ------------------------------------------------------------- */

