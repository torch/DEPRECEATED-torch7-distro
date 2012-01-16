#include "TH.h"
#include "luaT.h"
#include "utils.h"

#include "sys/time.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)

static const void* torch_ByteTensor_id;
static const void* torch_CharTensor_id;
static const void* torch_ShortTensor_id;
static const void* torch_IntTensor_id;
static const void* torch_LongTensor_id;
static const void* torch_FloatTensor_id;
static const void* torch_DoubleTensor_id;

static const void* torch_LongStorage_id;


#include "TensorMathWrap.c"

#include "generic/TensorConv.c"
#include "THGenerateAllTypes.h"

#include "generic/TensorLapack.c"
#include "THGenerateFloatTypes.h"

void torch_TensorMath_init(lua_State *L)
{
  torch_ByteTensorMath_init(L);
  torch_CharTensorMath_init(L);
  torch_ShortTensorMath_init(L);
  torch_IntTensorMath_init(L);
  torch_LongTensorMath_init(L);
  torch_FloatTensorMath_init(L);
  torch_DoubleTensorMath_init(L);

  luaL_register(L, NULL, torch_TensorMath__);
}
