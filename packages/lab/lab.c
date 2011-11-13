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
LUAT_DYNT_FUNCTION_WRAPPER(lab, max_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, max)
LUAT_DYNT_FUNCTION_WRAPPER(lab, min_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, min)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sum_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sum)
LUAT_DYNT_FUNCTION_WRAPPER(lab, prod_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, prod)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cumsum_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cumsum)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cumprod_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cumprod)
LUAT_DYNT_FUNCTION_WRAPPER(lab, trace)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cross_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cross)
LUAT_DYNT_FUNCTION_WRAPPER(lab, zeros_)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, zeros)
LUAT_DYNT_FUNCTION_WRAPPER(lab, ones_)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, ones)
LUAT_DYNT_FUNCTION_WRAPPER(lab, diag_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, diag)
LUAT_DYNT_FUNCTION_WRAPPER(lab, eye_)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, eye)
LUAT_DYNT_FUNCTION_WRAPPER(lab, range_)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, range)
LUAT_DYNT_FUNCTION_WRAPPER(lab, randperm_)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, randperm)
LUAT_DYNT_FUNCTION_WRAPPER(lab, reshape_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, reshape)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sort_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sort)
LUAT_DYNT_FUNCTION_WRAPPER(lab, tril_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, tril)
LUAT_DYNT_FUNCTION_WRAPPER(lab, triu_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, triu)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cat_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cat)
LUAT_DYNT_FUNCTION_WRAPPER(lab, conv2)
LUAT_DYNT_FUNCTION_WRAPPER(lab, xcorr2)
LUAT_DYNT_FUNCTION_WRAPPER(lab, conv3)
LUAT_DYNT_FUNCTION_WRAPPER(lab, xcorr3)

LUAT_DYNT_FUNCTION_WRAPPER(lab, log_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, log)
LUAT_DYNT_FUNCTION_WRAPPER(lab, log1p_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, log1p)
LUAT_DYNT_FUNCTION_WRAPPER(lab, exp_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, exp)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cos_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cos)
LUAT_DYNT_FUNCTION_WRAPPER(lab, acos_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, acos)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cosh_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, cosh)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sin_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sin)
LUAT_DYNT_FUNCTION_WRAPPER(lab, asin_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, asin)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sinh_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sinh)
LUAT_DYNT_FUNCTION_WRAPPER(lab, tan_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, tan)
LUAT_DYNT_FUNCTION_WRAPPER(lab, atan_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, atan)
LUAT_DYNT_FUNCTION_WRAPPER(lab, tanh_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, tanh)
LUAT_DYNT_FUNCTION_WRAPPER(lab, pow_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, pow)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sqrt_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, sqrt)
LUAT_DYNT_FUNCTION_WRAPPER(lab, ceil_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, ceil)
LUAT_DYNT_FUNCTION_WRAPPER(lab, floor_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, floor)
LUAT_DYNT_FUNCTION_WRAPPER(lab, abs_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, abs)
LUAT_DYNT_FUNCTION_WRAPPER(lab, mean_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, mean)
LUAT_DYNT_FUNCTION_WRAPPER(lab, std_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, std)
LUAT_DYNT_FUNCTION_WRAPPER(lab, var_)
LUAT_DYNT_FUNCTION_WRAPPER(lab, var)
LUAT_DYNT_FUNCTION_WRAPPER(lab, norm)
LUAT_DYNT_FUNCTION_WRAPPER(lab, dist)
LUAT_DYNT_FUNCTION_WRAPPER(lab, linspace_)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, linspace)
LUAT_DYNT_FUNCTION_WRAPPER(lab, logspace_)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, logspace)
LUAT_DYNT_FUNCTION_WRAPPER(lab, rand_)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, rand)
LUAT_DYNT_FUNCTION_WRAPPER(lab, randn_)
LUAT_DYNT_CONSTRUCTOR_WRAPPER(lab, randn)

static const struct luaL_Reg lab_stuff__ [] = {
  {"setdefaulttensortype", lab_setdefaulttensortype},
  {"getdefaulttensortype", lab_getdefaulttensortype},
  {"tic", lab_tic},
  {"toc", lab_toc},

  {"numel", lab_numel},
  //{"max_", lab_max_},
  {"max", lab_max},
  //{"min_", lab_min_},
  {"min", lab_min},
  //{"sum_", lab_sum_},
  {"sum", lab_sum},
  //{"prod_", lab_prod_},
  {"prod", lab_prod},
  //{"cumsum_", lab_cumsum_},
  {"cumsum", lab_cumsum},
  //{"cumprod_", lab_cumprod_},
  {"cumprod", lab_cumprod},
  {"trace", lab_trace},
  //{"cross_", lab_cross_},
  {"cross", lab_cross},
  //{"zeros_", lab_zeros_},
  {"zeros", lab_zeros},
  //{"ones_", lab_ones_},
  {"ones", lab_ones},
  //{"diag_", lab_diag_},
  {"diag", lab_diag},
  //{"eye_", lab_eye_},
  {"eye", lab_eye},
  //{"range_", lab_range_},
  {"range", lab_range},
  //{"randperm_", lab_randperm_},
  {"randperm", lab_randperm},
  //{"reshape_", lab_reshape_},
  {"reshape", lab_reshape},
  //{"sort_", lab_sort_},
  {"sort", lab_sort},
  //{"tril_", lab_tril_},
  {"tril", lab_tril},
  //{"triu_", lab_triu_},
  {"triu", lab_triu},
  //{"cat_", lab_cat_},
  {"cat", lab_cat},
  {"conv2", lab_conv2},
  {"xcorr2", lab_xcorr2},
  {"conv3", lab_conv3},
  {"xcorr3", lab_xcorr3},

  //{"log_", lab_log_},
  {"log", lab_log},
  //{"log1p_", lab_log1p_},
  {"log1p", lab_log1p},
  //{"exp_", lab_exp_},
  {"exp", lab_exp},
  //{"cos_", lab_cos_},
  {"cos", lab_cos},
  //{"acos_", lab_acos_},
  {"acos", lab_acos},
  //{"cosh_", lab_cosh_},
  {"cosh", lab_cosh},
  //{"sin_", lab_sin_},
  {"sin", lab_sin},
  //{"asin_", lab_asin_},
  {"asin", lab_asin},
  //{"sinh_", lab_sinh_},
  {"sinh", lab_sinh},
  //{"tan_", lab_tan_},
  {"tan", lab_tan},
  //{"atan_", lab_atan_},
  {"atan", lab_atan},
  //{"tanh_", lab_tanh_},
  {"tanh", lab_tanh},
  //{"pow_", lab_pow_},
  {"pow", lab_pow},
  //{"sqrt_", lab_sqrt_},
  {"sqrt", lab_sqrt},
  //{"ceil_", lab_ceil_},
  {"ceil", lab_ceil},
  //{"floor_", lab_floor_},
  {"floor", lab_floor},
  //{"abs_", lab_abs_},
  {"abs", lab_abs},
  //{"mean_", lab_mean_},
  {"mean", lab_mean},
  //{"std_", lab_std_},
  {"std", lab_std},
  //{"var_", lab_var_},
  {"var", lab_var},
  {"norm", lab_norm},
  {"dist", lab_dist},
  //{"linspace_", lab_linspace_},
  {"linspace", lab_linspace},
  //{"logspace_", lab_logspace_},
  {"logspace", lab_logspace},
  //{"rand_", lab_rand_},
  {"rand", lab_rand},
  //{"randn_", lab_randn_},
  {"randn", lab_randn},
  {NULL, NULL}
};

void lab_init(lua_State *L)
{
  luaL_register(L, NULL, lab_stuff__);
  lab_default_tensor_id = torch_DoubleTensor_id; /* obviously, this must be done after the "generic" stuff */
}
