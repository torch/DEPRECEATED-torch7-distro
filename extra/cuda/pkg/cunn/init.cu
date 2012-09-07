#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

const void *torch_CudaTensor_id = NULL;

#include "HardTanh.cu"
#include "Tanh.cu"
#include "LogSoftMax.cu"
#include "TemporalConvolution.cu"
#include "SpatialConvolution.cu"
#include "SpatialConvolutionMap.cu"
#include "SpatialSubSampling.cu"
#include "Square.cu"
#include "Sqrt.cu"
#include "MultiMarginCriterion.cu"
#include "MSECriterion.cu"

DLL_EXPORT TH_API int luaopen_libcunn(lua_State *L)
{
  lua_newtable(L);

  torch_CudaTensor_id = luaT_checktypename2id(L, "torch.CudaTensor");

  cunn_Tanh_init(L);
  cunn_HardTanh_init(L);
  cunn_LogSoftMax_init(L);
  cunn_TemporalConvolution_init(L);
  cunn_SpatialConvolution_init(L);
  cunn_SpatialConvolutionMap_init(L);
  cunn_SpatialSubSampling_init(L);
  cunn_MultiMarginCriterion_init(L);
  cunn_Square_init(L);
  cunn_Sqrt_init(L);
  cunn_MSECriterion_init(L);

  return 1;
}
