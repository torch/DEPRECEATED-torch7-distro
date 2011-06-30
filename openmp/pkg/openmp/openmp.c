#include "luaT.h"
#include "omp.h"

static int openmp_getNumThreads(lua_State* L)
{
  #pragma omp parallel
  if (omp_get_thread_num() == 0)
    lua_pushinteger(L, omp_get_num_threads());
  return 1;
}

static int openmp_setNumThreads(lua_State* L)
{
  int nth = luaL_checkint(L,1);
  omp_set_num_threads(nth);
  return 0;
}

static const struct luaL_Reg openmp_stuff__ [] = {
  {"getNumThreads", openmp_getNumThreads},
  {"setNumThreads", openmp_setNumThreads},
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
