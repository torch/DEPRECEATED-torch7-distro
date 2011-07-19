#include "general.h"

#ifdef _MSC_VER
#include <time.h>
#else
#include <sys/time.h>
#include <sys/resource.h>
#endif

#ifdef _MSC_VER
static time_t base_time = 0;
#endif    

static const void* torch_Timer_id = NULL;

typedef struct _Timer
{
    int isRunning;
    double totalrealtime;
    double totalcputime;
    double startrealtime;
    double startcputime;
} Timer;

static double torch_Timer_cputime()
{
#ifdef _MSC_VER
#error "not defined yet"
  time_t truc_foireux;
  time(&truc_foireux);
  return(difftime(truc_foireux, base_time));
#else
  struct rusage current;
  getrusage(RUSAGE_SELF, &current);
  return (current.ru_utime.tv_sec + current.ru_utime.tv_usec/1000000.0);
#endif
}

static double torch_Timer_realtime()
{
  struct timeval current;
  gettimeofday(&current, NULL);
  return (current.tv_sec + current.tv_usec/1000000.0);
}

static int torch_Timer_new(lua_State *L)
{
  Timer *timer = luaT_alloc(L, sizeof(Timer));
#ifdef _MSC_VER
  while(!base_time)
    time(&base_time);
#endif
  timer->isRunning = 1;
  timer->totalrealtime = 0;
  timer->totalcputime = 0;
  timer->startrealtime = torch_Timer_realtime();
  timer->startcputime = torch_Timer_cputime();
  luaT_pushudata(L, timer, torch_Timer_id);
  return 1;
}

static int torch_Timer_reset(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, torch_Timer_id);
  timer->totalrealtime = 0;
  timer->totalcputime = 0;
  timer->startrealtime = torch_Timer_realtime();
  timer->startcputime = torch_Timer_cputime();
  lua_settop(L, 1);
  return 1;
}

static int torch_Timer_free(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, torch_Timer_id);
  luaT_free(L, timer);
  return 0;
}

static int torch_Timer_stop(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, torch_Timer_id);
  if(timer->isRunning)  
  {
    double realtime = torch_Timer_realtime() - timer->startrealtime;
    double cputime = torch_Timer_cputime() - timer->startcputime;
    timer->totalrealtime += realtime;
    timer->totalcputime += cputime;
    timer->isRunning = 0;
  }
  lua_settop(L, 1);
  return 1;  
}

static int torch_Timer_resume(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, torch_Timer_id);
  if(!timer->isRunning)
  {
    timer->isRunning = 1;
    timer->startrealtime = torch_Timer_realtime();
    timer->startcputime = torch_Timer_cputime();
  }
  lua_settop(L, 1);
  return 1;  
}

static int torch_Timer_time(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, torch_Timer_id);
  double realtime = (timer->isRunning ? (timer->totalrealtime + torch_Timer_realtime() - timer->startrealtime) : timer->totalrealtime);
  double cputime = (timer->isRunning ? (timer->totalcputime + torch_Timer_cputime() - timer->startcputime) : timer->totalcputime);
  lua_pushnumber(L, cputime);
  lua_pushnumber(L, realtime);
  return 2;
}

static int torch_Timer___tostring__(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, torch_Timer_id);
  lua_pushfstring(L, "torch.Timer [status: %s]", (timer->isRunning ? "running" : "stopped"));
  return 1;
}

static const struct luaL_Reg torch_Timer__ [] = {
  {"reset", torch_Timer_reset},
  {"stop", torch_Timer_stop},
  {"resume", torch_Timer_resume},
  {"time", torch_Timer_time},
  {"__tostring__", torch_Timer___tostring__},
  {NULL, NULL}
};

void torch_Timer_init(lua_State *L)
{
  torch_Timer_id = luaT_newmetatable(L, "torch.Timer", NULL, torch_Timer_new, torch_Timer_free, NULL);
  luaL_register(L, NULL, torch_Timer__);
  lua_pop(L, 1);
}
