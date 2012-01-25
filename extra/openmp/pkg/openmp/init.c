#include "TH.h"
#include "luaT.h"
#include "omp.h"

int nthread;

int getdefaultnthread()
{
  return nthread;
}
int setdefaultnthread(int nthread_)
{
  nthread = nthread_;
  return nthread;
}

void setompnthread(lua_State *L, int ud, const char *field)
{
  /* handle number of threads */
  lua_getfield(L, ud, "nThread");
  if(lua_isnil(L, -1))
    omp_set_num_threads(getdefaultnthread());
  else
    omp_set_num_threads((int)lua_tointeger(L, -1));
}

extern void openmp_init(lua_State *L);
extern void torchOmp_init(lua_State *L);
extern void nnOmp_init(lua_State *L);

DLL_EXPORT int luaopen_libopenmp(lua_State *L)
{

  setdefaultnthread(omp_get_max_threads());
  openmp_init(L);
  torchOmp_init(L);
  nnOmp_init(L);

  return 1;
}
