#include "TensorMath.h"

static const void* torch_Tensor_id;

#define TENSOR_IMPLEMENT_BASIC_WRAPPER(FUNC) \
static int torch_Tensor_##FUNC(lua_State *L) \
{ \
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id); \
  THTensor_##FUNC(tensor); \
  lua_settop(L, 1); \
  return 1; \
}

TENSOR_IMPLEMENT_BASIC_WRAPPER(log)
TENSOR_IMPLEMENT_BASIC_WRAPPER(log1p)
TENSOR_IMPLEMENT_BASIC_WRAPPER(exp)
TENSOR_IMPLEMENT_BASIC_WRAPPER(cos)
TENSOR_IMPLEMENT_BASIC_WRAPPER(acos)
TENSOR_IMPLEMENT_BASIC_WRAPPER(cosh)
TENSOR_IMPLEMENT_BASIC_WRAPPER(sin)
TENSOR_IMPLEMENT_BASIC_WRAPPER(asin)
TENSOR_IMPLEMENT_BASIC_WRAPPER(sinh)
TENSOR_IMPLEMENT_BASIC_WRAPPER(tan)
TENSOR_IMPLEMENT_BASIC_WRAPPER(atan)
TENSOR_IMPLEMENT_BASIC_WRAPPER(tanh)
TENSOR_IMPLEMENT_BASIC_WRAPPER(sqrt)
TENSOR_IMPLEMENT_BASIC_WRAPPER(ceil)
TENSOR_IMPLEMENT_BASIC_WRAPPER(floor)
TENSOR_IMPLEMENT_BASIC_WRAPPER(abs)
TENSOR_IMPLEMENT_BASIC_WRAPPER(zero)

static int torch_Tensor_pow(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  double value = luaL_checknumber(L, 2);
  THTensor_pow(tensor, value);
  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_add(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  THTensor *src = NULL;
  int n = lua_gettop(L);

  if( (n == 2) && lua_isnumber(L, 2) )
  {
    double value = luaL_checknumber(L, 2);
    THTensor_add(tensor, value);
  }
  else if( (n == 2) && (src = luaT_toudata(L, 2, torch_Tensor_id)) )
    THTensor_addTensor(tensor, 1, src);
  else if( (n == 3) && lua_isnumber(L, 2) && (src = luaT_toudata(L, 3, torch_Tensor_id)) )
  {
    double value = luaL_checknumber(L, 2);
    THTensor_addTensor(tensor, value, src);
  }
  else
    luaL_error(L, "bad arguments: number, torch.Tensor, or number and torch.Tensor expected");

  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_mul(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  double value = luaL_checknumber(L, 2);
  THTensor_mul(tensor, value);
  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_cmul(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  THTensor *src = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor_cmul(tensor, src);
  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_addcmul(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  double value = luaL_checknumber(L, 2);
  THTensor *src1 = luaT_checkudata(L, 3, torch_Tensor_id);
  THTensor *src2 = luaT_checkudata(L, 4, torch_Tensor_id);
  THTensor_addcmul(tensor, value, src1, src2);
  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_div(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  double value = luaL_checknumber(L, 2);
  THTensor_div(tensor, value);
  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_cdiv(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  THTensor *src = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor_cdiv(tensor, src);
  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_addcdiv(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  double value = luaL_checknumber(L, 2);
  THTensor *src1 = luaT_checkudata(L, 3, torch_Tensor_id);
  THTensor *src2 = luaT_checkudata(L, 4, torch_Tensor_id);
  THTensor_addcdiv(tensor, value, src1, src2);
  lua_settop(L, 1);
  return 1;
}

/* statistics */

#define TENSOR_IMPLEMENT_BASIC_WRAPPER_STAT(FUNC) \
static int torch_Tensor_##FUNC(lua_State *L) \
{ \
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id); \
  lua_pushnumber(L, THTensor_##FUNC(tensor)); \
  return 1; \
}

TENSOR_IMPLEMENT_BASIC_WRAPPER_STAT(min)
TENSOR_IMPLEMENT_BASIC_WRAPPER_STAT(max)
TENSOR_IMPLEMENT_BASIC_WRAPPER_STAT(sum)
TENSOR_IMPLEMENT_BASIC_WRAPPER_STAT(mean)
TENSOR_IMPLEMENT_BASIC_WRAPPER_STAT(var)
TENSOR_IMPLEMENT_BASIC_WRAPPER_STAT(std)

static int torch_Tensor_norm(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  double value = luaL_optnumber(L, 2, 2);
  lua_pushnumber(L, THTensor_norm(tensor, value));
  return 1;
}

static int torch_Tensor_dist(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  THTensor *src = luaT_checkudata(L, 2, torch_Tensor_id);
  double value = luaL_optnumber(L, 3, 2);
  lua_pushnumber(L, THTensor_dist(tensor, src, value));
  return 1;
}

/* basic linear algebra */

static int torch_Tensor_dot(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  THTensor *src = luaT_checkudata(L, 2, torch_Tensor_id);
  lua_pushnumber(L, THTensor_dot(tensor, src));
  return 1;
}

#define TENSOR_IMPLEMENT_BASIC_ADDMUL(NAME) \
static int torch_Tensor_##NAME(lua_State *L) \
{ \
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id); \
  double alpha = luaL_checknumber(L, 2); \
  THTensor *src1 = luaT_checkudata(L, 3, torch_Tensor_id); \
  THTensor *src2 = luaT_checkudata(L, 4, torch_Tensor_id); \
\
  THTensor_##NAME(tensor, alpha, src1, src2); \
\
  lua_settop(L, 1); \
  return 1; \
}

TENSOR_IMPLEMENT_BASIC_ADDMUL(addT2dotT1)
TENSOR_IMPLEMENT_BASIC_ADDMUL(addT4dotT2)
TENSOR_IMPLEMENT_BASIC_ADDMUL(addT1outT1)
TENSOR_IMPLEMENT_BASIC_ADDMUL(addT2outT2)
TENSOR_IMPLEMENT_BASIC_ADDMUL(addT2dotT2)

static int torch_Tensor___add__(lua_State *L)
{
  THTensor *tensor1 = luaT_toudata(L, 1, torch_Tensor_id);
  THTensor *tensor2 = luaT_toudata(L, 2, torch_Tensor_id);
  THTensor *r;

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THTensor_new();
    luaT_pushudata(L, r, torch_Tensor_id);
    
    if(!tensor1 && tensor2)
    {
      THTensor_resizeAs(r, tensor2);
      THTensor_copy(r, tensor2);
      THTensor_add(r, luaL_checknumber(L, 1));
    }
    else if(tensor1 && !tensor2)
    {
      THTensor_resizeAs(r, tensor1);
      THTensor_copy(r, tensor1);
      THTensor_add(r, luaL_checknumber(L, 2));
    }
    else
    {
      THTensor_resizeAs(r, tensor1);
      THTensor_copy(r, tensor1);
      THTensor_addTensor(r, 1, tensor2);
    }
  }
  return 1;
}

static int torch_Tensor___sub__(lua_State *L)
{
  THTensor *tensor1 = luaT_toudata(L, 1, torch_Tensor_id);
  THTensor *tensor2 = luaT_toudata(L, 2, torch_Tensor_id);
  THTensor *r;

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THTensor_new();
    luaT_pushudata(L, r, torch_Tensor_id);
    
    if(!tensor1 && tensor2)
    {
      THTensor_resizeAs(r, tensor2);
      THTensor_fill(r, luaL_checknumber(L, 1));
      THTensor_addTensor(r, -1, tensor2);
    }
    else if(tensor1 && !tensor2)
    {
      THTensor_resizeAs(r, tensor1);
      THTensor_copy(r, tensor1);
      THTensor_add(r, -luaL_checknumber(L, 2));
    }
    else
    {
      THTensor_resizeAs(r, tensor1);
      THTensor_copy(r, tensor1);
      THTensor_addTensor(r, -1, tensor2);
    }
  }
  return 1;
}

static int torch_Tensor___unm__(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  THTensor *r;

  r = THTensor_new();
  luaT_pushudata(L, r, torch_Tensor_id);
  THTensor_resizeAs(r, tensor);
  THTensor_copy(r, tensor);
  THTensor_mul(r, -1);

  return 1;
}

static int torch_Tensor___mul__(lua_State *L)
{
  THTensor *tensor1 = luaT_toudata(L, 1, torch_Tensor_id);
  THTensor *tensor2 = luaT_toudata(L, 2, torch_Tensor_id);
  THTensor *r;

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THTensor_new();
    luaT_pushudata(L, r, torch_Tensor_id);
    
    if(!tensor1 && tensor2)
    {
      THTensor_resizeAs(r, tensor2);
      THTensor_copy(r, tensor2);
      THTensor_mul(r, luaL_checknumber(L, 1));
    }
    else if(tensor1 && !tensor2)
    {
      THTensor_resizeAs(r, tensor1);
      THTensor_copy(r, tensor1);
      THTensor_mul(r, luaL_checknumber(L, 2));
    }
    else
    {
      int dimt = tensor1->nDimension;
      int dims = tensor2->nDimension;
      
      if(dimt == 1 && dims == 1)
        lua_pushnumber(L, THTensor_dot(tensor1, tensor2)); /* ok, we wasted r, but who cares */
      else if(dimt == 2 && dims == 1)
      {
        THTensor_resize1d(r, tensor1->size[0]);
        THTensor_zero(r);
        THTensor_addT2dotT1(r, 1, tensor1, tensor2);
      }
      else if(dimt == 2 && dims == 2)
      {
        THTensor_resize2d(r, tensor1->size[0], tensor2->size[1]);
        THTensor_zero(r);
        THTensor_addT2dotT2(r, 1, tensor1, tensor2);
      }
      else if(dimt == 4 && dims == 2)
      {
        THTensor_resize2d(r, tensor1->size[0], tensor1->size[1]);
        THTensor_zero(r);
        THTensor_addT4dotT2(r, 1, tensor1, tensor2);
      }
      else
        luaL_error(L, "multiplication between %dD and %dD tensors not yet supported", tensor1->nDimension, tensor2->nDimension); 
    }
  }
  return 1;
}

static int torch_Tensor___div__(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  THTensor *r;

  luaL_argcheck(L, lua_isnumber(L,2), 2, "number expected");

  r = THTensor_new();
  luaT_pushudata(L, r, torch_Tensor_id);

  THTensor_resizeAs(r, tensor);
  THTensor_copy(r, tensor);
  THTensor_mul(r, 1/lua_tonumber(L, 2));

  return 1;
}

static const struct luaL_Reg torch_Tensor__ [] = {
  {"log", torch_Tensor_log},
  {"log1p", torch_Tensor_log1p},
  {"exp", torch_Tensor_exp},
  {"cos", torch_Tensor_cos},
  {"acos", torch_Tensor_acos},
  {"cosh", torch_Tensor_cosh},
  {"sin", torch_Tensor_sin},
  {"asin", torch_Tensor_asin},
  {"sinh", torch_Tensor_sinh},
  {"tan", torch_Tensor_tan},
  {"atan", torch_Tensor_atan},
  {"tanh", torch_Tensor_tanh},
  {"sqrt", torch_Tensor_sqrt},
  {"ceil", torch_Tensor_ceil},
  {"floor", torch_Tensor_floor},
  {"abs", torch_Tensor_abs},
  {"pow", torch_Tensor_pow},
  {"zero", torch_Tensor_zero},
  {"add", torch_Tensor_add},
  {"mul", torch_Tensor_mul},
  {"cmul", torch_Tensor_cmul},
  {"addcmul", torch_Tensor_addcmul},
  {"div", torch_Tensor_div},
  {"cdiv", torch_Tensor_cdiv},
  {"addcdiv", torch_Tensor_addcdiv},
  {"min", torch_Tensor_min},
  {"max", torch_Tensor_max},
  {"sum", torch_Tensor_sum},
  {"mean", torch_Tensor_mean},
  {"var", torch_Tensor_var},
  {"std", torch_Tensor_std},
  {"norm", torch_Tensor_norm},
  {"dist", torch_Tensor_dist},
  {"dot", torch_Tensor_dot},
  {"addT2dotT1", torch_Tensor_addT2dotT1},
  {"addT4dotT2", torch_Tensor_addT4dotT2},
  {"addT1outT1", torch_Tensor_addT1outT1},
  {"addT2outT2", torch_Tensor_addT2outT2},
  {"addT2dotT2", torch_Tensor_addT2dotT2},
  {"__add__", torch_Tensor___add__},
  {"__sub__", torch_Tensor___sub__},
  {"__unm__", torch_Tensor___unm__},
  {"__mul__", torch_Tensor___mul__},
  {"__div__", torch_Tensor___div__},
  {NULL, NULL}
};

void torch_TensorMath_init(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");

  luaT_pushmetaclass(L, torch_Tensor_id);
  luaL_register(L, NULL, torch_Tensor__);
  lua_pop(L, 1);
}
