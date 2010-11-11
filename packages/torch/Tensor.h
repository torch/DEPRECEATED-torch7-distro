#ifndef TORCH_TENSOR_INC
#define TORCH_TENSOR_INC

#include "general.h"

/* initializers. assume main table is on the stack */
void torch_ByteTensor_init(lua_State *L);
void torch_CharTensor_init(lua_State *L);
void torch_ShortTensor_init(lua_State *L);
void torch_IntTensor_init(lua_State *L);
void torch_LongTensor_init(lua_State *L);
void torch_FloatTensor_init(lua_State *L);
void torch_Tensor_init(lua_State *L);

#endif
