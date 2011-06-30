#include "TH.h"
#include "luaT.h"
#include "omp.h"

extern void openmp_init(lua_State *L);
extern void labOmp_init(lua_State *L);
extern void SpatialConvolutionOmp_init(lua_State *L);

DLL_EXPORT int luaopen_libopenmp(lua_State *L)
{

  openmp_init(L);
  labOmp_init(L);
  SpatialConvolutionOmp_init(L);

  return 1;
}
