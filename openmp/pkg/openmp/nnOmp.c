#include "TH.h"
#include "luaT.h"
#include "THOmpLabConv.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)
#define nnOmp_(NAME) TH_CONCAT_3(nnOmp_, Real, NAME)

static const void* torch_FloatTensor_id = NULL;
static const void* torch_DoubleTensor_id = NULL;

extern void setompnthread(lua_State *L, int ud, const char *field);

#include "generic/SpatialConvolutionOmp.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialConvolutionMapOmp.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialSubSamplingOmp.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialMaxPoolingOmp.c"
#include "THGenerateFloatTypes.h"

#include "generic/HardTanhOmp.c"
#include "THGenerateFloatTypes.h"

#include "generic/TanhOmp.c"
#include "THGenerateFloatTypes.h"

#include "generic/SqrtOmp.c"
#include "THGenerateFloatTypes.h"

#include "generic/SquareOmp.c"
#include "THGenerateFloatTypes.h"


DLL_EXPORT int nnOmp_init(lua_State *L)
{
  torch_FloatTensor_id = luaT_checktypename2id(L, "torch.FloatTensor");
  torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");

  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_getfield(L, LUA_GLOBALSINDEX, "nn");

  nnOmp_FloatSpatialConvolution_init(L);
  nnOmp_DoubleSpatialConvolution_init(L);

  nnOmp_FloatSpatialConvolutionMap_init(L);
  nnOmp_DoubleSpatialConvolutionMap_init(L);

  nnOmp_FloatSpatialSubSampling_init(L);
  nnOmp_DoubleSpatialSubSampling_init(L);

  nnOmp_FloatSpatialMaxPooling_init(L);
  nnOmp_DoubleSpatialMaxPooling_init(L);

  nnOmp_FloatHardTanh_init(L);
  nnOmp_DoubleHardTanh_init(L);

  nnOmp_FloatTanh_init(L);
  nnOmp_DoubleTanh_init(L);

  nnOmp_FloatSqrt_init(L);
  nnOmp_DoubleSqrt_init(L);

  nnOmp_FloatSquare_init(L);
  nnOmp_DoubleSquare_init(L);

  return 1;
}
