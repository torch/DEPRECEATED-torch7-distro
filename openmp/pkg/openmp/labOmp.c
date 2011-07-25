#include "TH.h"
#include "luaT.h"
#include "THOmpLabConv.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)

#define labOmp_(NAME) TH_CONCAT_3(labOmp_, Real, NAME)

static const void* torch_ByteTensor_id;
static const void* torch_CharTensor_id;
static const void* torch_ShortTensor_id;
static const void* torch_IntTensor_id;
static const void* torch_LongTensor_id;
static const void* torch_FloatTensor_id;
static const void* torch_DoubleTensor_id;

static const void* torch_LongStorage_id;

static const void* lab_default_tensor_id;

#include "generic/labOmp.c"
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


LUAT_DYNT_FUNCTION_WRAPPER(lab, conv2omp);
LUAT_DYNT_FUNCTION_WRAPPER(lab, xcorr2omp);

static const struct luaL_Reg labomp_stuff__ [] = {
  {"conv2omp", lab_conv2omp},
  {"xcorr2omp", lab_conv2omp},
  {NULL,NULL}
};

extern void labOmp_Byteinit(lua_State *L);
extern void labOmp_Charinit(lua_State *L);
extern void labOmp_Shortinit(lua_State *L);
extern void labOmp_Intinit(lua_State *L);
extern void labOmp_Longinit(lua_State *L);
extern void labOmp_Floatinit(lua_State *L);
extern void labOmp_Doubleinit(lua_State *L);

void labOmp_init(lua_State* L)
{
  labOmp_Byteinit(L);
  labOmp_Charinit(L);
  labOmp_Shortinit(L);
  labOmp_Intinit(L);
  labOmp_Longinit(L);
  labOmp_Floatinit(L);
  labOmp_Doubleinit(L);

  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_getfield(L, LUA_GLOBALSINDEX, "lab");
  luaL_register(L, NULL, labomp_stuff__);
  lua_pop(L,1);
}
