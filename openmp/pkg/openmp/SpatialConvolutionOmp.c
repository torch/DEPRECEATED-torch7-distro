#include "TH.h"
#include "luaT.h"
#include "THOmpLabConv.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)
#define nnOmp_(NAME) TH_CONCAT_3(nnOmp_, Real, NAME)

static const void* torch_FloatTensor_id = NULL;
static const void* torch_DoubleTensor_id = NULL;

#include "generic/SpatialConvolutionOmp.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int SpatialConvolutionOmp_init(lua_State *L)
{
  torch_FloatTensor_id = luaT_checktypename2id(L, "torch.FloatTensor");
  torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");

  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_getfield(L, LUA_GLOBALSINDEX, "nn");

  nnOmp_FloatSpatialConvolution_init(L);
  nnOmp_DoubleSpatialConvolution_init(L);

  return 1;
}
