#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/lab.c"
#else

/* note: debug the doacc (both begininng and end */
/* possibly put some defines */

LAB_IMPLEMENT_T(zero)

static int lab_(fill)(lua_State *L)
{
  if(lua_gettop(L) == 2)
  {
    THTensor *tensor = luaT_checkudata(L, 1, torch_(Tensor_id));
    real value = (real)luaL_checknumber(L, 2);
    THLab_(fill)(tensor, value);
  }
  else
    luaL_error(L, "invalid arguments: tensor number");

  return 1;
}

static int lab_(dot)(lua_State *L)
{
  if(lua_gettop(L) == 2)
  {
    THTensor *tensor = luaT_checkudata(L, 1, torch_(Tensor_id));
    THTensor *src = luaT_checkudata(L, 2, torch_(Tensor_id));
    lua_pushnumber(L, THLab_(dot)(tensor, src));
  }
  else
    luaL_error(L, "invalid arguments: tensor tensor");

  return 1;
}

LAB_IMPLEMENT_rNT(minall)
LAB_IMPLEMENT_rNT(maxall)
LAB_IMPLEMENT_rNT(sumall)

LAB_IMPLEMENT_oTTRoA(add)
LAB_IMPLEMENT_oTTRoA(mul)
LAB_IMPLEMENT_oTTRoA(div)

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


LAB_IMPLEMENT_oTTToA(cmul)
LAB_IMPLEMENT_oTTToA(cdiv)

LAB_IMPLEMENT_oTToRTToA(addcmul)
LAB_IMPLEMENT_oTToRTToA(addcdiv)

LAB_IMPLEMENT_oToRToRTToA(addmv)
LAB_IMPLEMENT_oToRToRTToA(addmm)
LAB_IMPLEMENT_oToRToRTToA(addr)

LAB_IMPLEMENT_rNT(numel)

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

LAB_IMPLEMENT_oTToI(sum)
LAB_IMPLEMENT_oTToI(prod)
LAB_IMPLEMENT_oTToI(cumsum)
LAB_IMPLEMENT_oTToI(cumprod)

LAB_IMPLEMENT_rNT(trace)


static int lab_(cross)(lua_State *L)
{
  THTensor *r_ = NULL, *a = NULL, *b = NULL;
  int dimension = 0;
  int narg = lua_gettop(L);

  if(narg == 4
     && luaT_isudata(L, 1, torch_(Tensor_id))
     && luaT_isudata(L, 2, torch_(Tensor_id))
     && luaT_isudata(L, 3, torch_(Tensor_id))
     && lua_isnumber(L, 4))
  {
    r_ = luaT_toudata(L, 1, torch_(Tensor_id));
    a = luaT_toudata(L, 2, torch_(Tensor_id));
    b = luaT_toudata(L, 3, torch_(Tensor_id));
    dimension = (int)(lua_tonumber(L, 4))-1;
  }
  else if(narg == 3
          && luaT_isudata(L, 1, torch_(Tensor_id))
          && luaT_isudata(L, 2, torch_(Tensor_id))
          && luaT_isudata(L, 3, torch_(Tensor_id)))
  {
    r_ = luaT_toudata(L, 1, torch_(Tensor_id));
    a = luaT_toudata(L, 2, torch_(Tensor_id));
    b = luaT_toudata(L, 3, torch_(Tensor_id));
  }
  else if(narg == 3
          && luaT_isudata(L, 1, torch_(Tensor_id))
          && luaT_isudata(L, 2, torch_(Tensor_id))
          && lua_isnumber(L, 3))
  {
    a = luaT_toudata(L, 1, torch_(Tensor_id));
    b = luaT_toudata(L, 2, torch_(Tensor_id));
    dimension = (int)(lua_tonumber(L, 3))-1;
  }
  else
    luaL_error(L, "invalid arguments: [tensor] tensor tensor [integer]");

  if(r_)
    THTensor_(retain)(r_);
  else
    r_ = THTensor_(new)();

  THLab_(cross)(r_, a, b, dimension);

  return 1;
}

LAB_IMPLEMENT_oTL(zeros)
LAB_IMPLEMENT_oTL(ones)

static int lab_(diag_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  long k = luaL_optnumber(L, 3, 0);

  THLab_(diag)(r_, t, k);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(diag)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 1 || (n == 2 && lua_type(L,2) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(diag_)(L);
}

static int lab_(eye_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  long n = luaL_checknumber(L, 2);
  long m = luaL_optnumber(L, 3, 0);

  THLab_(eye)(r_, n, m);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(eye)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 1 || (n == 2 && lua_type(L,1) == LUA_TNUMBER && lua_type(L,2) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(eye_)(L);
}

static int lab_(range_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  real xmin = luaL_checknumber(L, 2);
  real xmax = luaL_checknumber(L, 3);
  real step = luaL_optnumber(L, 4, 1);

  THLab_(range)(r_, xmin, xmax, step);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(range)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 2 || (n == 3 && lua_type(L,1) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(range_)(L);
}

static int lab_(randperm_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  long n = (long)luaL_checknumber(L, 2);

  THLab_(randperm)(r_, n);
  THLab_(add)(r_, r_, 1);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(randperm)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 1)
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  else if (n != 2 )
  {
    return luaL_error(L, "bad arguments: [result ,] n");
  }
  return lab_(randperm_)(L);
}

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

static int lab_(sort_)(lua_State *L)
{
  THTensor *rt_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THLongTensor *ri_ = luaT_checkudata(L, 2, torch_LongTensor_id);
  THTensor *t = luaT_checkudata(L, 3, torch_(Tensor_id));
  int dimension = luaL_optnumber(L, 4, THTensor_(nDimension)(t))-1;
  int descendingOrder = luaT_optboolean(L, 5, 0);

  THLab_(sort)(rt_, ri_, t, dimension, descendingOrder);
  THLongLab_add(ri_, ri_, 1);

  lua_settop(L, 2);
  return 2;
}

static int lab_(sort)(lua_State *L)
{
  int n = lua_gettop(L);
  if ( n == 1 || n == 2 || (n == 3 && lua_type(L,3) == LUA_TBOOLEAN))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
    luaT_pushudata(L, THLongTensor_new(), torch_LongTensor_id);
    lua_insert(L, 2);
  }
  return lab_(sort_)(L);
}

LAB_IMPLEMENT_oTToI(tril)
LAB_IMPLEMENT_oTToI(triu)


static int lab_(cat_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *ta = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *tb = luaT_checkudata(L, 3, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 4, 1))-1;

  THLab_(cat)(r_, ta, tb, dimension);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(cat)(lua_State *L)
{
  int n = lua_gettop(L);
  if ( n == 2 || (n == 3 && lua_type(L,3) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(cat_)(L);
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

LAB_IMPLEMENT_oTToI(mean)

static int lab_(std_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 3, THTensor_(nDimension)(t)))-1;
  int flag = luaT_optboolean(L, 4, 0);

  THLab_(std)(r_, t, dimension, flag);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(std)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 1 || 
      (n == 2 && lua_type(L,2) == LUA_TNUMBER) || 
      (n == 3 && lua_type(L,2) == LUA_TNUMBER && lua_type(L,3) == LUA_TBOOLEAN))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(std_)(L);
}

static int lab_(var_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 3, THTensor_(nDimension)(t)))-1;
  int flag = luaT_optboolean(L, 4, 0);

  THLab_(var)(r_, t, dimension, flag);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(var)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 1 || 
      (n == 2 && lua_type(L,2) == LUA_TNUMBER) || 
      (n == 3 && lua_type(L,2) == LUA_TNUMBER && lua_type(L,3) == LUA_TBOOLEAN))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(var_)(L);
}

static int lab_(norm)(lua_State *L)
{
  THTensor *t = luaT_checkudata(L, 1, torch_(Tensor_id));
  real value = luaL_optnumber(L, 2, 2);
  
  lua_pushnumber(L, THLab_(norm)(t, value));

  return 1;
}

static int lab_(dist)(lua_State *L)
{
  THTensor *t1 = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t2 = luaT_checkudata(L, 2, torch_(Tensor_id));
  real value = luaL_optnumber(L, 3, 2);
  
  lua_pushnumber(L, THLab_(dist)(t1, t2, value));

  return 1;
}

LAB_IMPLEMENT_rNT(meanall)
LAB_IMPLEMENT_rNT(varall)
LAB_IMPLEMENT_rNT(stdall)

static int lab_(linspace_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  real a = luaL_checknumber(L, 2);
  real b = luaL_checknumber(L, 3);
  long n = luaL_optnumber(L, 4, 100);

  THLab_(linspace)(r_, a, b, n);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(linspace)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 2 || (n == 3 && lua_type(L,1) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(linspace_)(L);
}

static int lab_(logspace_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  real a = luaL_checknumber(L, 2);
  real b = luaL_checknumber(L, 3);
  long n = luaL_optnumber(L, 4, 100);

  THLab_(logspace)(r_, a, b, n);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(logspace)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 2 || (n == 3 && lua_type(L,1) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(logspace_)(L);
}

LAB_IMPLEMENT_oTL(rand)
LAB_IMPLEMENT_oTL(randn)

LAB_IMPLEMENT_oTTxN(log)
LAB_IMPLEMENT_oTTxN(log1p)
LAB_IMPLEMENT_oTTxN(exp)
LAB_IMPLEMENT_oTTxN(cos)
LAB_IMPLEMENT_oTTxN(acos)
LAB_IMPLEMENT_oTTxN(cosh)
LAB_IMPLEMENT_oTTxN(sin)
LAB_IMPLEMENT_oTTxN(asin)
LAB_IMPLEMENT_oTTxN(sinh)
LAB_IMPLEMENT_oTTxN(tan)
LAB_IMPLEMENT_oTTxN(atan)
LAB_IMPLEMENT_oTTxN(tanh)
LAB_IMPLEMENT_oTTxN(sqrt)
LAB_IMPLEMENT_oTTxN(ceil)
LAB_IMPLEMENT_oTTxN(floor)
LAB_IMPLEMENT_oTTxN(abs)

LAB_IMPLEMENT_oTTNxNN(pow)

#endif

static const struct luaL_Reg lab_(stuff__) [] = {
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
  {"addmv", lab_(addmv)},
  {"maxall", lab_(maxall)},
  {"minall", lab_(minall)},
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
