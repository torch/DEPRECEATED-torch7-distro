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

extern void openmp_init(lua_State *L);
extern void labOmp_init(lua_State *L);
extern void nnOmp_init(lua_State *L);

DLL_EXPORT int luaopen_libopenmp(lua_State *L)
{

  setdefaultnthread(omp_get_max_threads());
  openmp_init(L);
  labOmp_init(L);
  nnOmp_init(L);

  return 1;
}
