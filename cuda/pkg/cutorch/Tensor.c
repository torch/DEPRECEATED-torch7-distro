#include "THC.h"
#include "THFile.h"
#include "luaT.h"

/* ids */
static const void *torch_File_id = NULL;

static const void *torch_ByteStorage_id = NULL;
static const void *torch_CharStorage_id = NULL;
static const void *torch_ShortStorage_id = NULL;
static const void *torch_IntStorage_id = NULL;
static const void *torch_LongStorage_id = NULL;
static const void *torch_FloatStorage_id = NULL;
static const void *torch_DoubleStorage_id = NULL;
static const void *torch_CudaStorage_id = NULL;

static const void *torch_ByteTensor_id = NULL;
static const void *torch_CharTensor_id = NULL;
static const void *torch_ShortTensor_id = NULL;
static const void *torch_IntTensor_id = NULL;
static const void *torch_LongTensor_id = NULL;
static const void *torch_FloatTensor_id = NULL;
static const void *torch_DoubleTensor_id = NULL;
static const void *torch_CudaTensor_id = NULL;

/* everything is as the generic Storage.c, except few things (see below) */

#define real float
#define Real Cuda
#define TH_GENERIC_FILE "generic/Tensor.c"

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,Real,Storage_,NAME)
#define torch_Storage_id TH_CONCAT_3(torch_,Real,Storage_id)
#define torch_Tensor_(NAME) TH_CONCAT_4(torch_,Real,Tensor_,NAME)
#define torch_Tensor_id TH_CONCAT_3(torch_,Real,Tensor_id)
#define STRING_torchTensor TH_CONCAT_STRING_3(torch.,Real,Tensor)

#include "generic/Tensor.c"

#undef real
#undef Real
#undef TH_GENERIC_FILE

/* now we overwrite some methods specific to CudaTensor */

#define CUDA_IMPLEMENT_TENSOR_COPY(TYPEC)                              \
  static int cutorch_##TYPEC##Tensor_copy(lua_State *L)                 \
  {                                                                     \
    TH##TYPEC##Tensor *storage = luaT_checkudata(L, 1, torch_##TYPEC##Tensor_id); \
    void *src;                                                          \
    if( (src = luaT_toudata(L, 2, torch_##TYPEC##Tensor_id)) )          \
      TH##TYPEC##Tensor_copy(storage, src);                             \
    else if( (src = luaT_toudata(L, 2, torch_ByteTensor_id)) )          \
      TH##TYPEC##Tensor_copyByte(storage, src);                         \
    else if( (src = luaT_toudata(L, 2, torch_CharTensor_id)) )          \
      TH##TYPEC##Tensor_copyChar(storage, src);                         \
    else if( (src = luaT_toudata(L, 2, torch_ShortTensor_id)) )         \
      TH##TYPEC##Tensor_copyShort(storage, src);                        \
    else if( (src = luaT_toudata(L, 2, torch_IntTensor_id)) )           \
      TH##TYPEC##Tensor_copyInt(storage, src);                          \
    else if( (src = luaT_toudata(L, 2, torch_LongTensor_id)) )          \
      TH##TYPEC##Tensor_copyLong(storage, src);                         \
    else if( (src = luaT_toudata(L, 2, torch_FloatTensor_id)) )         \
      TH##TYPEC##Tensor_copyFloat(storage, src);                        \
    else if( (src = luaT_toudata(L, 2, torch_DoubleTensor_id)) )        \
      TH##TYPEC##Tensor_copyDouble(storage, src);                       \
    else if( (src = luaT_toudata(L, 2, torch_CudaTensor_id)) )          \
      TH##TYPEC##Tensor_copyCuda(storage, src);                         \
    else                                                                \
      luaL_typerror(L, 2, "torch.*Tensor");                             \
                                                                        \
    lua_settop(L, 1);                                                   \
    return 1;                                                           \
  }

CUDA_IMPLEMENT_TENSOR_COPY(Byte)
CUDA_IMPLEMENT_TENSOR_COPY(Char)
CUDA_IMPLEMENT_TENSOR_COPY(Short)
CUDA_IMPLEMENT_TENSOR_COPY(Int)
CUDA_IMPLEMENT_TENSOR_COPY(Long)
CUDA_IMPLEMENT_TENSOR_COPY(Float)
CUDA_IMPLEMENT_TENSOR_COPY(Double)
CUDA_IMPLEMENT_TENSOR_COPY(Cuda)

static int cuda_CudaTensor_fill(lua_State *L)
{
  THCudaTensor *tensor = luaT_checkudata(L, 1, torch_CudaTensor_id);
  float value = (float)luaL_checknumber(L, 2);
  THCudaTensor_fill(tensor, value);
  lua_settop(L, 1);
  return 1;
}

void cutorch_CudaTensor_init(lua_State* L)
{
  /* the ids */
  torch_ByteTensor_id = luaT_checktypename2id(L, "torch.ByteTensor");
  torch_CharTensor_id = luaT_checktypename2id(L, "torch.CharTensor");
  torch_ShortTensor_id = luaT_checktypename2id(L, "torch.ShortTensor");
  torch_IntTensor_id = luaT_checktypename2id(L, "torch.IntTensor");
  torch_LongTensor_id = luaT_checktypename2id(L, "torch.LongTensor");
  torch_FloatTensor_id = luaT_checktypename2id(L, "torch.FloatTensor");
  torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");
  
  /* the standard stuff */
  torch_CudaTensor_init(L);

  /* additional methods */
  luaT_pushmetaclass(L, torch_CudaTensor_id);
  lua_pushcfunction(L, cuda_CudaTensor_fill);
  lua_setfield(L, -2, "fill");
  lua_pop(L, 1);

  /* the copy methods */
  {
    int i;

    const void* ids[8] = {torch_ByteTensor_id,
                          torch_CharTensor_id,
                          torch_ShortTensor_id,
                          torch_IntTensor_id,
                          torch_LongTensor_id,
                          torch_FloatTensor_id,
                          torch_DoubleTensor_id,
                          torch_CudaTensor_id};
    
    static int (*funcs[8])(lua_State*) = {cutorch_ByteTensor_copy,
                                          cutorch_CharTensor_copy,
                                          cutorch_ShortTensor_copy,
                                          cutorch_IntTensor_copy,
                                          cutorch_LongTensor_copy,
                                          cutorch_FloatTensor_copy,
                                          cutorch_DoubleTensor_copy,
                                          cutorch_CudaTensor_copy};

    for(i = 0; i < 8; i++)
    {
      luaT_pushmetaclass(L, ids[i]);
      lua_pushcfunction(L, funcs[i]);
      lua_setfield(L, -2, "copy");
      lua_pop(L, 1);
    }
  }
}
