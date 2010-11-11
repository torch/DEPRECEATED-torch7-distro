#include "Timer.h"

#ifdef _MSC_VER
#include <time.h>
#else
#include <sys/times.h>
#include <unistd.h>
#endif

#ifdef _MSC_VER
static time_t base_time = 0;
#endif    

static const void* torch_Timer_id = NULL;

typedef struct _Timer
{
    int isRunning;
    double totalTime;
    double startTime;
} Timer;

static double torch_Timer_runTime()
{
#ifdef _MSC_VER
  time_t truc_foireux;
  time(&truc_foireux);
  return(difftime(truc_foireux, base_time));
#else
  struct tms current;
  times(&current);
  return(((double)current.tms_utime)/((double)sysconf(_SC_CLK_TCK)));
#endif
}

static int torch_Timer_new(lua_State *L)
{
  Timer *timer = luaT_alloc(L, sizeof(Timer));
#ifdef _MSC_VER
  while(!base_time)
    time(&base_time);
#endif
  timer->totalTime = 0;
  timer->isRunning = 1;
  timer->startTime = torch_Timer_runTime();
  luaT_pushudata(L, timer, torch_Timer_id);
  return 1;
}

static int torch_Timer_reset(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, torch_Timer_id);
  timer->totalTime = 0;
  timer->startTime = torch_Timer_runTime();
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
    double currentTime = torch_Timer_runTime() - timer->startTime;
    timer->totalTime += currentTime;
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
    timer->startTime = torch_Timer_runTime();
    timer->isRunning = 1;
  }
  lua_settop(L, 1);
  return 1;  
}

static int torch_Timer_time(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, torch_Timer_id);
  double returnTime = (timer->isRunning ? (timer->totalTime + torch_Timer_runTime() - timer->startTime) : timer->totalTime);
  lua_pushnumber(L, returnTime);
  return 1;  
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
