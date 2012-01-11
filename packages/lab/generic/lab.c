#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/lab.c"
#else

#include "interfaces.c"

LAB_IMPLEMENT_oTTRoA(add)

static int lab_(cadd)(lua_State *L)
{
  THTensor *r_ = NULL, *t = NULL, *src = NULL;
  real value = 1;
  int narg = lua_gettop(L);
  int doacc = 0;

  if(narg > 0 && lua_isboolean(L, -1))
  {
    doacc = lua_toboolean(L, -1);
    lua_pop(L, 1);
    narg--;
  }

  if(narg == 4
     && luaT_isudata(L, 1, torch_(Tensor_id))
     && luaT_isudata(L, 2, torch_(Tensor_id))
     && lua_isnumber(L, 3)
     && luaT_isudata(L, 4, torch_(Tensor_id)))
  {
    r_ = luaT_toudata(L, 1, torch_(Tensor_id));
    t = luaT_toudata(L, 2, torch_(Tensor_id));
    value = lua_tonumber(L, 3);
    src = luaT_toudata(L, 4, torch_(Tensor_id));
  }
  else if(narg == 3
          && luaT_isudata(L, 1, torch_(Tensor_id))
          && luaT_isudata(L, 2, torch_(Tensor_id))
          && luaT_isudata(L, 3, torch_(Tensor_id)))
  {
    r_ = luaT_toudata(L, 1, torch_(Tensor_id));
    t = luaT_toudata(L, 2, torch_(Tensor_id));
    src = luaT_toudata(L, 3, torch_(Tensor_id));
  }
  else if(narg == 3
          && luaT_isudata(L, 1, torch_(Tensor_id))
          && lua_isnumber(L, 2)
          && luaT_isudata(L, 3, torch_(Tensor_id)))
  {
    t = luaT_toudata(L, 1, torch_(Tensor_id));
    value = lua_tonumber(L, 2);
    src = luaT_toudata(L, 3, torch_(Tensor_id));
  }
  else
    THError("invalid arguments: [result] tensor [number] tensor");

  if(!r_)
  {
    if(doacc)
      r_ = t;
    else
        r_ = THTensor_(new)();
  }
  else
    THTensor_(retain)(r_);

  luaT_pushudata(L, r_, torch_(Tensor_id));

  THLab_(cadd)(r_, t, value, src);

  return 1;
}

#define LAB_IMPLEMENT_MINMAX(NAME)                                      \
  static int lab_(NAME)(lua_State *L)                                   \
  {                                                                     \
    THTensor *values = NULL, *t = NULL;                                 \
    THLongTensor *indices = NULL;                                       \
    int dimension = 0; /* DEBUG: a voir */                              \
    int narg = lua_gettop(L);                                           \
                                                                        \
    if(narg == 4                                                        \
       && luaT_isudata(L, 1, torch_(Tensor_id))                         \
       && luaT_isudata(L, 2, torch_LongTensor_id)                       \
       && luaT_isudata(L, 3, torch_(Tensor_id))                         \
       && lua_isnumber(L, 4))                                           \
    {                                                                   \
      values = luaT_toudata(L, 1, torch_(Tensor_id));                   \
      indices = luaT_toudata(L, 2, torch_LongTensor_id);                \
      t = luaT_toudata(L, 3, torch_(Tensor_id));                        \
      dimension = (int)lua_tonumber(L, 4)-1;                            \
    }                                                                   \
    else if(narg == 3                                                   \
            && luaT_isudata(L, 1, torch_(Tensor_id))                    \
            && luaT_isudata(L, 2, torch_LongTensor_id)                  \
            && luaT_isudata(L, 3, torch_(Tensor_id)))                   \
    {                                                                   \
      values = luaT_toudata(L, 1, torch_(Tensor_id));                   \
      indices = luaT_toudata(L, 2, torch_LongTensor_id);                \
      t = luaT_toudata(L, 3, torch_(Tensor_id));                        \
    }                                                                   \
    else if(narg == 2                                                   \
            && luaT_isudata(L, 1, torch_(Tensor_id))                    \
            && lua_isnumber(L, 2))                                      \
    {                                                                   \
      t = luaT_toudata(L, 1, torch_(Tensor_id));                        \
      dimension = (int)lua_tonumber(L, 2)-1;                            \
    }                                                                   \
    else if(narg == 1                                                   \
            && luaT_isudata(L, 1, torch_(Tensor_id)))                   \
    {                                                                   \
      t = luaT_toudata(L, 1, torch_(Tensor_id));                        \
    }                                                                   \
    else                                                                \
      luaL_error(L, "invalid arguments: [tensor longtensor] tensor [dimension]"); \
                                                                        \
    if(values)                                                          \
    {                                                                   \
      THTensor_(retain)(values);                                        \
      THLongTensor_retain(indices);                                     \
    }                                                                   \
    else                                                                \
    {                                                                   \
      values = THTensor_(new)();                                        \
      indices = THLongTensor_new();                                     \
    }                                                                   \
                                                                        \
    luaT_pushudata(L, values, torch_(Tensor_id));                       \
    luaT_pushudata(L, indices, torch_LongTensor_id);                    \
                                                                        \
    THLab_(NAME)(values, indices, t, dimension);                        \
    THLongLab_add(indices, indices, 1);                                 \
                                                                        \
    return 2;                                                           \
  }

LAB_IMPLEMENT_MINMAX(min)
LAB_IMPLEMENT_MINMAX(max)


LAB_IMPLEMENT_oTL(zeros)
LAB_IMPLEMENT_oTL(ones)

static int lab_(reshape_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  THLongStorage *dimension = lab_checklongargs(L, 3);

  THLab_(reshape)(r_, t, dimension);

  THLongStorage_free(dimension);
  lua_settop(L, 1);  
  return 1;
}

static int lab_(reshape)(lua_State *L)
{
  if (lua_type(L,2) == LUA_TNUMBER)
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(reshape_)(L);
}

static int lab_(histc)(lua_State *L)
{
  THTensor *r = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *h = luaT_checkudata(L, 2, torch_(Tensor_id));
  int nbins = luaL_checknumber(L, 3);
  real *h_data = THTensor_(data)(h);

  TH_TENSOR_APPLY(real, r,                                      \
                  if ((*r_data <= nbins) && (*r_data >= 1)) {   \
                    *(h_data + (int)(*r_data) - 1) += 1;        \
                  })
  return 0;
}

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

LAB_IMPLEMENT_oTL(rand)
LAB_IMPLEMENT_oTL(randn)

#endif

static const struct luaL_Reg lab_(stuff__) [] = {
  {"zero", lab_(zero)},
  {"fill", lab_(fill)},
  {"dot", lab_(dot)},
  {"minall", lab_(minall)},
  {"sumall", lab_(sumall)},
  {"maxall", lab_(maxall)},
  {"mul", lab_(mul)},
  {"div", lab_(div)},
  {"cmul", lab_(cmul)},
  {"cdiv", lab_(cdiv)},
  {"addcmul", lab_(addcmul)},
  {"addcdiv", lab_(addcdiv)},
  {"addmv", lab_(addmv)},
  {"addmm", lab_(addmm)},
  {"addr", lab_(addr)},
  {"numel", lab_(numel)},
  {"max", lab_(max)},
  {"min", lab_(min)},
  {"sum", lab_(sum)},
  {"prod", lab_(prod)},
  {"cumsum", lab_(cumsum)},
  {"cumprod", lab_(cumprod)},
  {"trace", lab_(trace)},
  {"cross", lab_(cross)},
  {"zeros", lab_(zeros)},
  {"ones", lab_(ones)},
  {"diag", lab_(diag)},
  {"eye", lab_(eye)},
  {"range", lab_(range)},
  {"randperm", lab_(randperm)},
  {"reshape_", lab_(reshape_)},
  {"reshape", lab_(reshape)},
  {"sort", lab_(sort)},
  {"tril", lab_(tril)},
  {"triu", lab_(triu)},
  {"_histc", lab_(histc)},
  {"cat", lab_(cat)},
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"log", lab_(log)},
  {"log1p", lab_(log1p)},
  {"exp", lab_(exp)},
  {"cos", lab_(cos)},
  {"acos", lab_(acos)},
  {"cosh", lab_(cosh)},
  {"sin", lab_(sin)},
  {"asin", lab_(asin)},
  {"sinh", lab_(sinh)},
  {"tan", lab_(tan)},
  {"atan", lab_(atan)},
  {"tanh", lab_(tanh)},
  {"pow", lab_(pow)},
  {"sqrt", lab_(sqrt)},
  {"ceil", lab_(ceil)},
  {"floor", lab_(floor)},
  {"abs", lab_(abs)},
  {"mean", lab_(mean)},
  {"std", lab_(std)},
  {"var", lab_(var)},
  {"norm", lab_(norm)},
  {"dist", lab_(dist)},
  {"meanall", lab_(meanall)},
  {"varall", lab_(varall)},
  {"stdall", lab_(stdall)},
  {"linspace", lab_(linspace)},
  {"logspace", lab_(logspace)},
  {"rand", lab_(rand)},
  {"randn", lab_(randn)},
#endif
  {NULL, NULL}
};

void lab_(init)(lua_State *L)
{
  torch_(Tensor_id) = luaT_checktypename2id(L, torch_string_(Tensor));
  torch_LongStorage_id = luaT_checktypename2id(L, "torch.LongStorage");

  /* register everything into the "lab" field of the tensor metaclass */
  luaT_pushmetaclass(L, torch_(Tensor_id));
  lua_pushstring(L, "lab");
  lua_newtable(L);
  luaL_register(L, NULL, lab_(stuff__));
  lua_rawset(L, -3);
  lua_pop(L, 1);

/*  luaT_registeratid(L, lab_(stuff__), torch_(Tensor_id)); */
/*  luaL_register(L, NULL, lab_(stuff__)); */  
}

#endif
