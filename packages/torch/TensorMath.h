#ifndef TORCH_TENSOR_MATH_INC
#define TORCH_TENSOR_MATH_INC

#include "Tensor.h"

void torch_ByteTensorMath_init(lua_State *L);
void torch_CharTensorMath_init(lua_State *L);
void torch_ShortTensorMath_init(lua_State *L);
void torch_IntTensorMath_init(lua_State *L);
void torch_LongTensorMath_init(lua_State *L);
void torch_FloatTensorMath_init(lua_State *L);
void torch_DoubleTensorMath_init(lua_State *L);

#endif
