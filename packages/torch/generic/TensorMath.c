#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TensorMath.c"
#else

static const void* torch_Tensor_id;

#define TENSOR_IMPLEMENT_BASIC_WRAPPER(FUNC)                    \
  static int torch_TensorMath_(FUNC)(lua_State *L)              \
  {                                                             \
    THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);  \
    THTensor_(FUNC)(tensor);                                    \
    lua_settop(L, 1);                                           \
    return 1;                                                   \
  }

#define TENSOR_IMPLEMENT_BASIC_WRAPPER_STAT(FUNC)               \
  static int torch_TensorMath_(FUNC)(lua_State *L)              \
  {                                                             \
    THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);  \
    lua_pushnumber(L, THTensor_(FUNC)(tensor));                 \
    return 1;                                                   \
  }

static int torch_TensorMath_(fill)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  real value = (real)luaL_checknumber(L, 2);
  THTensor_(fill)(tensor, value);
  lua_settop(L, 1);
  return 1;
}

TENSOR_IMPLEMENT_BASIC_WRAPPER(zero)

static int torch_TensorMath_(add)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  THTensor *src = NULL;
  int n = lua_gettop(L);

  if( (n == 2) && lua_isnumber(L, 2) )
  {
    double value = luaL_checknumber(L, 2);
    THTensor_(add)(tensor, value);
  }
  else if( (n == 2) && (src = luaT_toudata(L, 2, torch_Tensor_id)) )
    THTensor_(cadd)(tensor, 1, src);
  else if( (n == 3) && lua_isnumber(L, 2) && (src = luaT_toudata(L, 3, torch_Tensor_id)) )
  {
    double value = luaL_checknumber(L, 2);
    THTensor_(cadd)(tensor, value, src);
  }
  else
    luaL_error(L, "bad arguments: number, torch.Tensor, or number and torch.Tensor expected");

  lua_settop(L, 1);
  return 1;
}

static int torch_TensorMath_(mul)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  double value = luaL_checknumber(L, 2);
  THTensor_(mul)(tensor, value);
  lua_settop(L, 1);
  return 1;
}

static int torch_TensorMath_(cmul)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  THTensor *src = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor_(cmul)(tensor, src);
  lua_settop(L, 1);
  return 1;
}

static int torch_TensorMath_(addcmul)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  double value = luaL_checknumber(L, 2);
  THTensor *src1 = luaT_checkudata(L, 3, torch_Tensor_id);
  THTensor *src2 = luaT_checkudata(L, 4, torch_Tensor_id);
  THTensor_(addcmul)(tensor, value, src1, src2);
  lua_settop(L, 1);
  return 1;
}

static int torch_TensorMath_(div)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  double value = luaL_checknumber(L, 2);
  THTensor_(div)(tensor, value);
  lua_settop(L, 1);
  return 1;
}

static int torch_TensorMath_(cdiv)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  THTensor *src = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor_(cdiv)(tensor, src);
  lua_settop(L, 1);
  return 1;
}

static int torch_TensorMath_(addcdiv)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  double value = luaL_checknumber(L, 2);
  THTensor *src1 = luaT_checkudata(L, 3, torch_Tensor_id);
  THTensor *src2 = luaT_checkudata(L, 4, torch_Tensor_id);
  THTensor_(addcdiv)(tensor, value, src1, src2);
  lua_settop(L, 1);
  return 1;
}

static int torch_TensorMath_(dot)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  THTensor *src = luaT_checkudata(L, 2, torch_Tensor_id);
  lua_pushnumber(L, THTensor_(dot)(tensor, src));
  return 1;
}

TENSOR_IMPLEMENT_BASIC_WRAPPER_STAT(min)
TENSOR_IMPLEMENT_BASIC_WRAPPER_STAT(max)
TENSOR_IMPLEMENT_BASIC_WRAPPER_STAT(sum)

#define TENSOR_IMPLEMENT_BASIC_ADDMUL(NAME)                     \
  static int torch_TensorMath_(NAME)(lua_State *L)                  \
  {                                                             \
    THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);  \
    double alpha = luaL_checknumber(L, 2);                      \
    THTensor *src1 = luaT_checkudata(L, 3, torch_Tensor_id);    \
    THTensor *src2 = luaT_checkudata(L, 4, torch_Tensor_id);    \
                                                                \
    THTensor_(NAME)(tensor, alpha, src1, src2);                 \
                                                                \
    lua_settop(L, 1);                                           \
    return 1;                                                   \
}

#define TENSOR_IMPLEMENT_BASIC_BETA_ADDMUL(NAME)                    \
  static int torch_TensorMath_(NAME)(lua_State *L)                  \
  {                                                                 \
    THTensor *tensor = NULL, *src1 = NULL, *src2 = NULL;            \
    double alpha = 1, beta = 1;                                     \
    int narg = lua_gettop(L);                                       \
                                                                    \
    if(narg == 4)                                                   \
    {                                                               \
      tensor = luaT_checkudata(L, 1, torch_Tensor_id);              \
      alpha = luaL_checknumber(L, 2);                               \
      src1 = luaT_checkudata(L, 3, torch_Tensor_id);                \
      src2 = luaT_checkudata(L, 4, torch_Tensor_id);                \
    }                                                               \
    else if(narg == 5)                                              \
    {                                                               \
      tensor = luaT_checkudata(L, 1, torch_Tensor_id);              \
      beta = luaL_checknumber(L, 2);                                \
      alpha = luaL_checknumber(L, 3);                               \
      src1 = luaT_checkudata(L, 4, torch_Tensor_id);                \
      src2 = luaT_checkudata(L, 5, torch_Tensor_id);                \
    }                                                               \
    else                                                            \
      luaL_error(L, "expected arguments: tensor, [beta], alpha, tensor, tensor"); \
                                                                        \
    THTensor_(NAME)(tensor, beta, alpha, src1, src2);                   \
                                                                        \
    lua_settop(L, 1);                                                   \
    return 1;                                                           \
  }

TENSOR_IMPLEMENT_BASIC_BETA_ADDMUL(addmv)
TENSOR_IMPLEMENT_BASIC_BETA_ADDMUL(addmm)
TENSOR_IMPLEMENT_BASIC_ADDMUL(addr)

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

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

static int torch_TensorMath_(pow)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  double value = luaL_checknumber(L, 2);
  THTensor_(pow)(tensor, value);
  lua_settop(L, 1);
  return 1;
}

TENSOR_IMPLEMENT_BASIC_WRAPPER(sqrt)
TENSOR_IMPLEMENT_BASIC_WRAPPER(ceil)
TENSOR_IMPLEMENT_BASIC_WRAPPER(floor)
TENSOR_IMPLEMENT_BASIC_WRAPPER(abs)

TENSOR_IMPLEMENT_BASIC_WRAPPER_STAT(mean)
TENSOR_IMPLEMENT_BASIC_WRAPPER_STAT(var)
TENSOR_IMPLEMENT_BASIC_WRAPPER_STAT(std)

static int torch_TensorMath_(norm)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  double value = luaL_optnumber(L, 2, 2);
  lua_pushnumber(L, THTensor_(norm)(tensor, value));
  return 1;
}

static int torch_TensorMath_(dist)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  THTensor *src = luaT_checkudata(L, 2, torch_Tensor_id);
  double value = luaL_optnumber(L, 3, 2);
  lua_pushnumber(L, THTensor_(dist)(tensor, src, value));
  return 1;
}

#endif

static int torch_TensorMath_(__add__)(lua_State *L)
{
  THTensor *tensor1 = luaT_toudata(L, 1, torch_Tensor_id);
  THTensor *tensor2 = luaT_toudata(L, 2, torch_Tensor_id);
  THTensor *r;

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THTensor_(new)();
    luaT_pushudata(L, r, torch_Tensor_id);
    
    if(!tensor1 && tensor2)
    {
      THTensor_(resizeAs)(r, tensor2);
      THTensor_(copy)(r, tensor2);
      THTensor_(add)(r, luaL_checknumber(L, 1));
    }
    else if(tensor1 && !tensor2)
    {
      THTensor_(resizeAs)(r, tensor1);
      THTensor_(copy)(r, tensor1);
      THTensor_(add)(r, luaL_checknumber(L, 2));
    }
    else
    {
      THTensor_(resizeAs)(r, tensor1);
      THTensor_(copy)(r, tensor1);
      THTensor_(cadd)(r, 1, tensor2);
    }
  }
  return 1;
}

static int torch_TensorMath_(__sub__)(lua_State *L)
{
  THTensor *tensor1 = luaT_toudata(L, 1, torch_Tensor_id);
  THTensor *tensor2 = luaT_toudata(L, 2, torch_Tensor_id);
  THTensor *r;

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THTensor_(new)();
    luaT_pushudata(L, r, torch_Tensor_id);
    
    if(!tensor1 && tensor2)
    {
      THTensor_(resizeAs)(r, tensor2);
      THTensor_(fill)(r, luaL_checknumber(L, 1));
      THTensor_(cadd)(r, -1, tensor2);
    }
    else if(tensor1 && !tensor2)
    {
      THTensor_(resizeAs)(r, tensor1);
      THTensor_(copy)(r, tensor1);
      THTensor_(add)(r, -luaL_checknumber(L, 2));
    }
    else
    {
      THTensor_(resizeAs)(r, tensor1);
      THTensor_(copy)(r, tensor1);
      THTensor_(cadd)(r, -1, tensor2);
    }
  }
  return 1;
}

static int torch_TensorMath_(__unm__)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  THTensor *r;

  r = THTensor_(new)();
  luaT_pushudata(L, r, torch_Tensor_id);
  THTensor_(resizeAs)(r, tensor);
  THTensor_(copy)(r, tensor);
  THTensor_(mul)(r, -1);

  return 1;
}

static int torch_TensorMath_(__mul__)(lua_State *L)
{
  THTensor *tensor1 = luaT_toudata(L, 1, torch_Tensor_id);
  THTensor *tensor2 = luaT_toudata(L, 2, torch_Tensor_id);
  THTensor *r;

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THTensor_(new)();
    luaT_pushudata(L, r, torch_Tensor_id);
    
    if(!tensor1 && tensor2)
    {
      THTensor_(resizeAs)(r, tensor2);
      THTensor_(copy)(r, tensor2);
      THTensor_(mul)(r, luaL_checknumber(L, 1));
    }
    else if(tensor1 && !tensor2)
    {
      THTensor_(resizeAs)(r, tensor1);
      THTensor_(copy)(r, tensor1);
      THTensor_(mul)(r, luaL_checknumber(L, 2));
    }
    else
    {
      int dimt = tensor1->nDimension;
      int dims = tensor2->nDimension;
      
      if(dimt == 1 && dims == 1)
        lua_pushnumber(L, THTensor_(dot)(tensor1, tensor2)); /* ok, we wasted r, but who cares */
      else if(dimt == 2 && dims == 1)
      {
        THTensor_(resize1d)(r, tensor1->size[0]);
        THTensor_(zero)(r);
        THTensor_(addmv)(r, 1, 1, tensor1, tensor2);
      }
      else if(dimt == 2 && dims == 2)
      {
        THTensor_(resize2d)(r, tensor1->size[0], tensor2->size[1]);
        THTensor_(zero)(r);
        THTensor_(addmm)(r, 1, 1, tensor1, tensor2);
      }
      else
        luaL_error(L, "multiplication between %dD and %dD tensors not yet supported", tensor1->nDimension, tensor2->nDimension); 
    }
  }
  return 1;
}

static int torch_TensorMath_(__div__)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor_id);
  THTensor *r;

  luaL_argcheck(L, lua_isnumber(L,2), 2, "number expected");

  r = THTensor_(new)();
  luaT_pushudata(L, r, torch_Tensor_id);

  THTensor_(resizeAs)(r, tensor);
  THTensor_(copy)(r, tensor);
  THTensor_(mul)(r, 1/lua_tonumber(L, 2));

  return 1;
}

static const struct luaL_Reg torch_TensorMath_(_) [] = {
  {"fill", torch_TensorMath_(fill)},
  {"zero", torch_TensorMath_(zero)},
  {"add", torch_TensorMath_(add)},
  {"mul", torch_TensorMath_(mul)},
  {"cmul", torch_TensorMath_(cmul)},
  {"addcmul", torch_TensorMath_(addcmul)},
  {"div", torch_TensorMath_(div)},
  {"cdiv", torch_TensorMath_(cdiv)},
  {"addcdiv", torch_TensorMath_(addcdiv)},
  {"dot", torch_TensorMath_(dot)},
  {"min", torch_TensorMath_(min)},
  {"max", torch_TensorMath_(max)},
  {"sum", torch_TensorMath_(sum)},
  {"addmv", torch_TensorMath_(addmv)},
  {"addmm", torch_TensorMath_(addmm)},
  {"addr", torch_TensorMath_(addr)},

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"log", torch_TensorMath_(log)},
  {"log1p", torch_TensorMath_(log1p)},
  {"exp", torch_TensorMath_(exp)},
  {"cos", torch_TensorMath_(cos)},
  {"acos", torch_TensorMath_(acos)},
  {"cosh", torch_TensorMath_(cosh)},
  {"sin", torch_TensorMath_(sin)},
  {"asin", torch_TensorMath_(asin)},
  {"sinh", torch_TensorMath_(sinh)},
  {"tan", torch_TensorMath_(tan)},
  {"atan", torch_TensorMath_(atan)},
  {"tanh", torch_TensorMath_(tanh)},
  {"pow", torch_TensorMath_(pow)},
  {"sqrt", torch_TensorMath_(sqrt)},
  {"ceil", torch_TensorMath_(ceil)},
  {"floor", torch_TensorMath_(floor)},
  {"abs", torch_TensorMath_(abs)},

  {"mean", torch_TensorMath_(mean)},
  {"var", torch_TensorMath_(var)},
  {"std", torch_TensorMath_(std)},
  {"norm", torch_TensorMath_(norm)},
  {"dist", torch_TensorMath_(dist)},
#endif

  {"__add__", torch_TensorMath_(__add__)},
  {"__sub__", torch_TensorMath_(__sub__)},
  {"__unm__", torch_TensorMath_(__unm__)},
  {"__mul__", torch_TensorMath_(__mul__)},
  {"__div__", torch_TensorMath_(__div__)},
  {NULL, NULL}
};

void torch_TensorMath_(init)(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, STRING_torchTensor);

  luaT_pushmetaclass(L, torch_Tensor_id);
  luaL_register(L, NULL, torch_TensorMath_(_));
  lua_pop(L, 1);
}

#endif
