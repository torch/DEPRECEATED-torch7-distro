#include "luaT.h"
#include "omp.h"

extern int getdefaultnthread();
extern int setdefaultnthread(int);

static int openmp_getNumThreads(lua_State* L)
{
  lua_pushinteger(L, omp_get_max_threads());
  return 1;
}

static int openmp_setNumThreads(lua_State* L)
{
  int nth = luaL_checkint(L,1);
  omp_set_num_threads(nth);
  return 0;
}

static int openmp_getDefaultNumThreads(lua_State* L)
{
  lua_pushinteger(L,getdefaultnthread());
  return 1;
}
static int openmp_setDefaultNumThreads(lua_State* L)
{
  int nth = luaL_checkint(L,1);
  lua_pushinteger(L,setdefaultnthread(nth));
  return 1;
}


static const struct luaL_Reg openmp_stuff__ [] = {
  {"getNumThreads", openmp_getNumThreads},
  {"setNumThreads", openmp_setNumThreads},
  {"getDefaultNumThreads", openmp_getDefaultNumThreads},
  {"setDefaultNumThreads", openmp_setDefaultNumThreads},
  {NULL,NULL}
};

void openmp_init(lua_State* L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "openmp");
  luaL_register(L, NULL, openmp_stuff__);
  lua_pop(L,1);
}
