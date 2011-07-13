#include "THC.h"
#include "luaT.h"

static const void *torch_CudaStorage_id = NULL;

#define real float
#define Real Cuda

#define torch_TensorMath_(NAME) TH_CONCAT_4(torch_,Real,TensorMath_,NAME)
#define torch_Tensor_id TH_CONCAT_3(torch_,Real,Tensor_id)
#define STRING_torchTensor TH_CONCAT_STRING_3(torch.,Real,Tensor)

#define TH_GENERIC_FILE "generic/TensorMath.c"
#include "generic/TensorMath.c"
#undef TH_GENERIC_FILE

void cutorch_CudaTensorMath_init(lua_State* L)
{
  torch_CudaTensorMath_init(L);
}
