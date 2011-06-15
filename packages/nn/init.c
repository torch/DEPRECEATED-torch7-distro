#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

static const void* torch_FloatTensor_id = NULL;
static const void* torch_DoubleTensor_id = NULL;

#include "generic/HardTanh.c"
#include "THGenerateFloatTypes.h"

#include "generic/Exp.c"
#include "THGenerateFloatTypes.h"

#include "generic/LogSigmoid.c"
#include "THGenerateFloatTypes.h"

#include "generic/LogSoftmax.c"
#include "THGenerateFloatTypes.h"

#include "generic/Sigmoid.c"
#include "THGenerateFloatTypes.h"

#include "generic/SoftPlus.c"
#include "THGenerateFloatTypes.h"

#include "generic/Tanh.c"
#include "THGenerateFloatTypes.h"

#include "generic/SoftMax.c"
#include "THGenerateFloatTypes.h"

#include "generic/Mean.c"
#include "THGenerateFloatTypes.h"

#include "generic/Max.c"
#include "THGenerateFloatTypes.h"

#include "generic/Min.c"
#include "THGenerateFloatTypes.h"

#include "generic/Sum.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int luaopen_libnn(lua_State *L)
{
  torch_FloatTensor_id = luaT_checktypename2id(L, "torch.FloatTensor");
  torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");

  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "nn");

  nn_FloatMean_init(L);
  nn_FloatMin_init(L);
  nn_FloatMax_init(L);
  nn_FloatSum_init(L);
  nn_FloatExp_init(L);
  nn_FloatHardTanh_init(L);
  nn_FloatLogSoftMax_init(L);

  nn_DoubleMean_init(L);
  nn_DoubleMin_init(L);
  nn_DoubleMax_init(L);
  nn_DoubleSum_init(L);
  nn_DoubleExp_init(L);
  nn_DoubleHardTanh_init(L);
  nn_DoubleLogSoftMax_init(L);

/*  
  nn_FloatLogSigmoid_init(L);
  nn_Sigmoid_init(L);
  nn_SoftMax_init(L);
  nn_SoftPlus_init(L);
  nn_Tanh_init(L);

  nn_SpatialConvolution_init(L);
  nn_SpatialSubSampling_init(L);
  nn_TemporalConvolution_init(L);
  nn_TemporalSubSampling_init(L);
  nn_SparseLinear_init(L);
  nn_MSECriterion_init(L);
  nn_AbsCriterion_init(L);
*/

  return 1;
}
