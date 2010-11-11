#ifndef TORCH_STORAGE_INC
#define TORCH_STORAGE_INC

#include "general.h"

/* initializers. assume main table is on the stack */
void torch_ByteStorage_init(lua_State *L);
void torch_CharStorage_init(lua_State *L);
void torch_ShortStorage_init(lua_State *L);
void torch_IntStorage_init(lua_State *L);
void torch_LongStorage_init(lua_State *L);
void torch_FloatStorage_init(lua_State *L);
void torch_DoubleStorage_init(lua_State *L);

#endif
