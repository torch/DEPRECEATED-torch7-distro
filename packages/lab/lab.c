#include "TH.h"
#include "luaT.h"
#include "utils.h"

#include "sys/time.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)

#define lab_(NAME) TH_CONCAT_3(lab_, Real, NAME)

static const void* torch_ByteTensor_id;
static const void* torch_CharTensor_id;
static const void* torch_ShortTensor_id;
static const void* torch_IntTensor_id;
static const void* torch_LongTensor_id;
static const void* torch_FloatTensor_id;
static const void* torch_DoubleTensor_id;

static const void* torch_LongStorage_id;

static const void* lab_default_tensor_id;

#include "generic/lab.c"
#include "THGenerateAllTypes.h"

#include "generic/labconv.c"
#include "THGenerateAllTypes.h"

#include "generic/lablapack.c"
#include "THGenerateFloatTypes.h"

static int lab_setdefaulttensortype(lua_State *L)
{
  const void *id;

  luaL_checkstring(L, 1);
  
  if(!(id = luaT_typename2id(L, lua_tostring(L, 1))))                  \
    return luaL_error(L, "<%s> is not a string describing a torch object", lua_tostring(L, 1)); \

  lab_default_tensor_id = id;

  return 0;
}

static int lab_getdefaulttensortype(lua_State *L)
{
  lua_pushstring(L, luaT_id2typename(L, lab_default_tensor_id));
  return 1;
}

static int lab_tic(lua_State* L)
{
  struct timeval tv;
  gettimeofday(&tv,NULL);
  double ttime = (double)tv.tv_sec + (double)(tv.tv_usec)/1000000.0;
  lua_pushnumber(L,ttime);
  return 1;
}

static int lab_toc(lua_State* L)
{
  struct timeval tv;
  gettimeofday(&tv,NULL);
  double toctime = (double)tv.tv_sec + (double)(tv.tv_usec)/1000000.0;
  lua_Number tictime = luaL_checknumber(L,1);
  lua_pushnumber(L,toctime-tictime);
  return 1;
}



#define LUAT_DYNT_FUNCTION_WRAPPER(PKG, FUNC)                           \
  static int PKG##_##FUNC(lua_State *L)                                 \
  {                                                                     \
    if(!luaT_getmetaclass(L, 1))                                        \
    {                                                                   \
      const void *id = lab_default_tensor_id;                           \
      luaT_pushmetaclass(L, id);                                        \
    }                                                                   \
                                                                        \
    lua_pushstring(L, #PKG);                                            \
    lua_rawget(L, -2);                                                  \
    if(lua_istable(L, -1))                                              \
    {                                                                   \
      lua_pushstring(L, #FUNC);                                         \
      lua_rawget(L, -2);                                                \
      if(lua_isfunction(L, -1))                                         \
      {                                                                 \
        lua_insert(L, 1);                                               \
        lua_pop(L, 2); /* the two tables we put on the stack above */   \
        lua_call(L, lua_gettop(L)-1, LUA_MULTRET);                      \
      }                                                                 \
      else                                                              \
        return luaL_error(L, "%s does not implement the " #PKG "." #FUNC "() function", luaT_typename(L, 1)); \
    }                                                                   \
    else                                                                \
      return luaL_error(L, "%s does not implement " #PKG " functions", luaT_typename(L, 1)); \
                                                                        \
    return lua_gettop(L);                                               \
  }

#define LUAT_DYNT_CONSTRUCTOR_WRAPPER(PKG, FUNC)                        \
  static int PKG##_##FUNC(lua_State *L)                                 \
  {                                                                     \
    const void* id;                                                     \
    if(lua_isstring(L, -1) && (!lua_isnumber(L, -1)))                   \
    {                                                                   \
      if(!(id = luaT_typename2id(L, lua_tostring(L, -1))))              \
        return luaL_error(L, "<%s> is not string describing a torch object", lua_tostring(L, -1)); \
      lua_pop(L, 1); /* pop the class name */                           \
    }                                                                   \
    else                                                                \
      id = lab_default_tensor_id;                                       \
                                                                        \
    luaT_pushmetaclass(L, id);                                          \
    lua_pushstring(L, #PKG);                                            \
    lua_rawget(L, -2);                                                  \
    if(lua_istable(L, -1))                                              \
    {                                                                   \
      lua_pushstring(L, #FUNC);                                         \
      lua_rawget(L, -2);                                                \
      if(lua_isfunction(L, -1))                                         \
        {                                                               \
          lua_insert(L, 1);                                             \
          lua_pop(L, 2); /* the two tables we put on the stack above */ \
          lua_call(L, lua_gettop(L)-1, LUA_MULTRET);                    \
        }                                                               \
      else                                                              \
        return luaL_error(L, "%s does not implement the " #PKG "." #FUNC "() function", luaT_id2typename(L, id)); \
    }                                                                   \
    else                                                                \
      return luaL_error(L, "%s does not implement " #PKG " functions", luaT_id2typename(L, id)); \
                                                                        \
    return lua_gettop(L);                                               \
  }

LUAT_DYNT_FUNCTION_WRAPPER(lab, numel)
LUAT_DYNT_FUNCTION_WRAPPER(lab, max)
LUAT_DYNT_FUNCTION_WRAPPER(lab, maxall)
LUAT_DYNT_FUNCTION_WRAPPER(lab, minall)
LUAT_DYNT_FUNCTION_WRAPPER(lab, min)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sum)
LUAT_DYNT_FUNCTION_WRAPPER(lab, prod)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cumsum)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cumprod)
LUAT_DYNT_FUNCTION_WRAPPER(lab, trace)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cross)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, zeros)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, ones)
LUAT_DYNT_FUNCTION_WRAPPER(lab, diag)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, eye)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, range)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, randperm)
LUAT_DYNT_FUNCTION_WRAPPER(lab, reshape)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sort)
LUAT_DYNT_FUNCTION_WRAPPER(lab, tril)
LUAT_DYNT_FUNCTION_WRAPPER(lab, triu)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cat)
LUAT_DYNT_FUNCTION_WRAPPER(lab, conv2)
LUAT_DYNT_FUNCTION_WRAPPER(lab, xcorr2)
LUAT_DYNT_FUNCTION_WRAPPER(lab, conv3)
LUAT_DYNT_FUNCTION_WRAPPER(lab, xcorr3)

LUAT_DYNT_FUNCTION_WRAPPER(lab, addmv)

LUAT_DYNT_FUNCTION_WRAPPER(lab, gesv)
LUAT_DYNT_FUNCTION_WRAPPER(lab, gels)
LUAT_DYNT_FUNCTION_WRAPPER(lab, eig)
LUAT_DYNT_FUNCTION_WRAPPER(lab, svd)
LUAT_DYNT_FUNCTION_WRAPPER(lab, log)
LUAT_DYNT_FUNCTION_WRAPPER(lab, log1p)
LUAT_DYNT_FUNCTION_WRAPPER(lab, exp)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cos)
LUAT_DYNT_FUNCTION_WRAPPER(lab, acos)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cosh)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sin)
LUAT_DYNT_FUNCTION_WRAPPER(lab, asin)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sinh)
LUAT_DYNT_FUNCTION_WRAPPER(lab, tan)
LUAT_DYNT_FUNCTION_WRAPPER(lab, atan)
LUAT_DYNT_FUNCTION_WRAPPER(lab, tanh)
LUAT_DYNT_FUNCTION_WRAPPER(lab, pow)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sqrt)
LUAT_DYNT_FUNCTION_WRAPPER(lab, ceil)
LUAT_DYNT_FUNCTION_WRAPPER(lab, floor)
LUAT_DYNT_FUNCTION_WRAPPER(lab, abs)
LUAT_DYNT_FUNCTION_WRAPPER(lab, mean)
LUAT_DYNT_FUNCTION_WRAPPER(lab, std)
LUAT_DYNT_FUNCTION_WRAPPER(lab, var)
LUAT_DYNT_FUNCTION_WRAPPER(lab, norm)
LUAT_DYNT_FUNCTION_WRAPPER(lab, dist)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, linspace)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, logspace)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, rand)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, randn)

static const struct luaL_Reg lab_stuff__ [] = {
  {"setdefaulttensortype", lab_setdefaulttensortype},
  {"getdefaulttensortype", lab_getdefaulttensortype},
  {"tic", lab_tic},
  {"toc", lab_toc},

  {"numel", lab_numel},
  {"max", lab_max},
  {"maxall", lab_maxall},
  {"minall", lab_minall},
  {"min", lab_min},
  {"sum", lab_sum},
  {"prod", lab_prod},
  {"cumsum", lab_cumsum},
  {"cumprod", lab_cumprod},
  {"trace", lab_trace},
  {"cross", lab_cross},
  {"zeros", lab_zeros},
  {"ones", lab_ones},
  {"diag", lab_diag},
  {"eye", lab_eye},
  {"range", lab_range},
  {"randperm", lab_randperm},
  {"reshape", lab_reshape},
  {"sort", lab_sort},
  {"tril", lab_tril},
  {"triu", lab_triu},
  {"cat", lab_cat},
  {"conv2", lab_conv2},
  {"xcorr2", lab_xcorr2},
  {"conv3", lab_conv3},
  {"xcorr3", lab_xcorr3},

  {"svd",  lab_svd},
  {"eig",  lab_eig},
  {"gels", lab_gels},
  {"gesv", lab_gesv},

  {"log", lab_log},
  {"log1p", lab_log1p},
  {"exp", lab_exp},
  {"cos", lab_cos},
  {"acos", lab_acos},
  {"cosh", lab_cosh},
  {"sin", lab_sin},
  {"asin", lab_asin},
  {"sinh", lab_sinh},
  {"tan", lab_tan},
  {"atan", lab_atan},
  {"tanh", lab_tanh},
  {"pow", lab_pow},
  {"sqrt", lab_sqrt},
  {"ceil", lab_ceil},
  {"floor", lab_floor},
  {"abs", lab_abs},
  {"mean", lab_mean},
  {"std", lab_std},
  {"var", lab_var},
  {"norm", lab_norm},
  {"dist", lab_dist},
  {"linspace", lab_linspace},
  {"logspace", lab_logspace},
  {"rand", lab_rand},
  {"randn", lab_randn},
  {"addmv", lab_addmv},
  {NULL, NULL}
};

void lab_init(lua_State *L)
{
  luaL_register(L, NULL, lab_stuff__);
  lab_default_tensor_id = torch_DoubleTensor_id; /* obviously, this must be done after the "generic" stuff */
}
