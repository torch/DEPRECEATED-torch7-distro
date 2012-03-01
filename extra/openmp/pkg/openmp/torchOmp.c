#include "TH.h"
#include "luaT.h"
#include "THOmpTensorConv.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)

#define torchOmp_(NAME) TH_CONCAT_3(torchOmp_, Real, NAME)

static const void* torch_ByteTensor_id;
static const void* torch_CharTensor_id;
static const void* torch_ShortTensor_id;
static const void* torch_IntTensor_id;
static const void* torch_LongTensor_id;
static const void* torch_FloatTensor_id;
static const void* torch_DoubleTensor_id;

static const void* torch_LongStorage_id;

#include "generic/torchOmp.c"
#include "THGenerateAllTypes.h"

#define LUAT_DYNT_FUNCTION_WRAPPER(PKG, FUNC)                           \
  static int PKG##_##FUNC(lua_State *L)                                 \
  {                                                                     \
    if(luaT_getmetaclass(L, 1))                                         \
    {                                                                   \
      lua_pushstring(L, #PKG);                                          \
      lua_rawget(L, -2);                                                \
      if(lua_istable(L, -1))                                            \
      {                                                                 \
        lua_pushstring(L, #FUNC);                                       \
        lua_rawget(L, -2);                                              \
        if(lua_isfunction(L, -1))                                       \
        {                                                               \
          lua_insert(L, 1);                                             \
          lua_pop(L, 2); /* the two tables we put on the stack above */ \
          lua_call(L, lua_gettop(L)-1, LUA_MULTRET);                    \
        }                                                               \
        else                                                            \
          return luaL_error(L, "%s does not implement the " #PKG "." #FUNC "() function", luaT_typename(L, 1)); \
      }                                                                 \
      else                                                              \
        return luaL_error(L, "%s does not implement " #PKG " functions", luaT_typename(L, 1)); \
    }                                                                   \
    else                                                                \
      return luaL_error(L, "first argument is not a torch object");     \
                                                                        \
    return lua_gettop(L);                                               \
  }


LUAT_DYNT_FUNCTION_WRAPPER(torch, conv2omp);
LUAT_DYNT_FUNCTION_WRAPPER(torch, xcorr2omp);

static const struct luaL_Reg torchomp_stuff__ [] = {
  {"conv2omp", torch_conv2omp},
  {"xcorr2omp", torch_conv2omp},
  {NULL,NULL}
};

extern void torchOmp_Byteinit(lua_State *L);
extern void torchOmp_Charinit(lua_State *L);
extern void torchOmp_Shortinit(lua_State *L);
extern void torchOmp_Intinit(lua_State *L);
extern void torchOmp_Longinit(lua_State *L);
extern void torchOmp_Floatinit(lua_State *L);
extern void torchOmp_Doubleinit(lua_State *L);

void torchOmp_init(lua_State* L)
{
  torchOmp_Byteinit(L);
  torchOmp_Charinit(L);
  torchOmp_Shortinit(L);
  torchOmp_Intinit(L);
  torchOmp_Longinit(L);
  torchOmp_Floatinit(L);
  torchOmp_Doubleinit(L);

  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_getfield(L, LUA_GLOBALSINDEX, "torch");
  luaL_register(L, NULL, torchomp_stuff__);
  lua_pop(L,1);
}
