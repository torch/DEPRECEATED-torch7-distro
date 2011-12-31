#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/lab.c"
#else

static int lab_(zero)(lua_State *L)
{
  if(lua_gettop(L) == 1)
  {
    THTensor *tensor = luaT_checkudata(L, 1, torch_(Tensor_id));
    THLab_(zero)(tensor);
  }
  else
    luaL_error(L, "invalid arguments: tensor");

  return 1;
}

static int lab_(fill)(lua_State *L)
{
  if(lua_gettop(L) == 2)
  {
    THTensor *tensor = luaT_checkudata(L, 1, torch_(Tensor_id));
    real value = (real)luaL_checknumber(L, 2);
    THLab_(fill)(tensor, value);
  }
  else
    luaL_error(L, "invalid arguments: tensor value");

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

static int lab_(minall)(lua_State *L)
{
  if(lua_gettop(L) == 1)
  {
    THTensor *tensor = luaT_checkudata(L, 1, torch_(Tensor_id));
    lua_pushnumber(L, THLab_(minall)(tensor));
  }
  else
    luaL_error(L, "invalid arguments: tensor");

  return 1;
}

static int lab_(maxall)(lua_State *L)
{
  if(lua_gettop(L) == 1)
  {
    THTensor *tensor = luaT_checkudata(L, 1, torch_(Tensor_id));
    lua_pushnumber(L, THLab_(maxall)(tensor));
  }
  else
    luaL_error(L, "invalid arguments: tensor");

  return 1;
}

static int lab_(sumall)(lua_State *L)
{
  if(lua_gettop(L) == 1)
  {
    THTensor *tensor = luaT_checkudata(L, 1, torch_(Tensor_id));
    lua_pushnumber(L, THLab_(sumall)(tensor));
  }
  else
    luaL_error(L, "invalid arguments: tensor");

  return 1;
}

#define LAB_IMPLEMENT_ADDMULDIV(NAME)                             \
  static int lab_(NAME)(lua_State *L)                             \
  {                                                               \
    THTensor *r_ = NULL, *t = NULL;                               \
    real value = 0;                                               \
    int narg = lua_gettop(L);                                     \
    int doacc = 0;                                                \
                                                                  \
    if(narg > 0 && lua_isboolean(L, -1))                          \
    {                                                             \
      doacc = lua_toboolean(L, -1);                               \
      lua_pop(L, 1);                                              \
      narg--;                                                     \
    }                                                             \
                                                                  \
    if(narg == 2                                                  \
       && luaT_isudata(L, 1, torch_(Tensor_id))                   \
       && lua_isnumber(L, 2))                                     \
    {                                                             \
      t = luaT_checkudata(L, 1, torch_(Tensor_id));               \
      value = luaL_checknumber(L, 2);                             \
    }                                                             \
    else if(narg == 3                                             \
            && luaT_isudata(L, 1, torch_(Tensor_id))              \
            && luaT_isudata(L, 2, torch_(Tensor_id))              \
            && lua_isnumber(L, 3))                                \
    {                                                             \
      r_ = luaT_checkudata(L, 1, torch_(Tensor_id));              \
      t = luaT_checkudata(L, 2, torch_(Tensor_id));               \
      value = luaL_checknumber(L, 3);                             \
    }                                                             \
    else                                                          \
      luaL_error(L, "invalid arguments: [tensor] tensor value");  \
                                                                  \
    if(!r_)                                                       \
    {                                                             \
      if(doacc)                                                   \
        r_ = t;                                                   \
      else                                                        \
        r_ = THTensor_(new)();                                    \
    }                                                             \
    else                                                          \
      THTensor_(retain)(r_);                                      \
                                                                  \
    luaT_pushudata(L, r_, torch_(Tensor_id));                     \
                                                                  \
    THLab_(NAME)(r_, t, value);                                   \
                                                                  \
    return 1;                                                     \
  }

LAB_IMPLEMENT_ADDMULDIV(add)
LAB_IMPLEMENT_ADDMULDIV(mul)
LAB_IMPLEMENT_ADDMULDIV(div)

static int lab_(cadd)(lua_State *L)
{
  THTensor *r_, *t, *src;
  real value;
  int n = lua_gettop(L);

  if(n == 4)
  {
    r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
    t = luaT_checkudata(L, 2, torch_(Tensor_id));
    value = luaL_checknumber(L, 3);
    src = luaT_checkudata(L, 4, torch_(Tensor_id));
  }
  else if(n == 3)
  {
    if(lua_isuserdata(L, 2))
    {
      r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
      t = luaT_checkudata(L, 2, torch_(Tensor_id));
      value = 1;
      src = luaT_checkudata(L, 3, torch_(Tensor_id));
    }
    else
    {
      r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
      t = r_; 
      value = luaL_checknumber(L, 2);
      src = luaT_checkudata(L, 3, torch_(Tensor_id));
    }
  }
  else
    THError("invalid number of arguments");

  THLab_(cadd)(r_, t, value, src);
  return 1;
}

static int lab_(cmul)(lua_State *L)
{
  THTensor *r_, *t, *src;
  int n = lua_gettop(L);

  if(n == 3)
  {
    r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
    t = luaT_checkudata(L, 2, torch_(Tensor_id));
    src = luaT_checkudata(L, 3, torch_(Tensor_id));
  }
  else if(n == 2)
  {
    r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
    t = r_; 
    src = luaT_checkudata(L, 2, torch_(Tensor_id));
  }
  else
    THError("invalid number of arguments");
  
  THLab_(cmul)(r_, t, src);
  return 1;
}

static int lab_(cdiv)(lua_State *L)
{
  THTensor *r_, *t, *src;
  int n = lua_gettop(L);

  if(n == 3)
  {
    r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
    t = luaT_checkudata(L, 2, torch_(Tensor_id));
    src = luaT_checkudata(L, 3, torch_(Tensor_id));
  }
  else if(n == 2)
  {
    r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
    t = r_; 
    src = luaT_checkudata(L, 2, torch_(Tensor_id));
  }
  else
    THError("invalid number of arguments");
  
  THLab_(cdiv)(r_, t, src);
  return 1;
}

static int lab_(addcmul)(lua_State *L)
{
  THTensor *r_, *t, *src1, *src2;
  real value;
  int n = lua_gettop(L);

  if(n == 5)
  {
    r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
    t = luaT_checkudata(L, 2, torch_(Tensor_id));
    value = luaL_checknumber(L, 3);
    src1 = luaT_checkudata(L, 4, torch_(Tensor_id));
    src2 = luaT_checkudata(L, 5, torch_(Tensor_id));
  }
  else if(n == 4)
  {
    if(lua_isuserdata(L, 2))
    {
      r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
      t = luaT_checkudata(L, 2, torch_(Tensor_id));
      value = 1;
      src1 = luaT_checkudata(L, 3, torch_(Tensor_id));
      src2 = luaT_checkudata(L, 4, torch_(Tensor_id));
    }
    else
    {
      r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
      t = r_;
      value = luaL_checknumber(L, 2);
      src1 = luaT_checkudata(L, 3, torch_(Tensor_id));
      src2 = luaT_checkudata(L, 4, torch_(Tensor_id));
    }
  }
  else
    THError("invalid number of arguments");
  
  THLab_(addcmul)(r_, t, value, src1, src2);
  return 1;
}

static int lab_(addcdiv)(lua_State *L)
{
  THTensor *r_ = NULL, *t, *src1, *src2;
  real value = 1;
  int n = lua_gettop(L);

  if(n == 5)
  {
    r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
    t = luaT_checkudata(L, 2, torch_(Tensor_id));
    value = luaL_checknumber(L, 3);
    src1 = luaT_checkudata(L, 4, torch_(Tensor_id));
    src2 = luaT_checkudata(L, 5, torch_(Tensor_id));
  }
  else if(n == 4)
  {
    if(lua_isuserdata(L, 2))
    {
      r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
      t = luaT_checkudata(L, 2, torch_(Tensor_id));
      src1 = luaT_checkudata(L, 3, torch_(Tensor_id));
      src2 = luaT_checkudata(L, 4, torch_(Tensor_id));
    }
    else
    {
      t = luaT_checkudata(L, 1, torch_(Tensor_id));
      value = luaL_checknumber(L, 2);
      src1 = luaT_checkudata(L, 3, torch_(Tensor_id));
      src2 = luaT_checkudata(L, 4, torch_(Tensor_id));
    }
  }
  else
    THError("invalid number of arguments");

  /* seul changement avec torch package... r_ = t */
  if(!r_)
    r_ = THTensor_(new)();

  THLab_(addcdiv)(r_, t, value, src1, src2);

  return 1;
}

#define LAB_IMPLEMENT_ADDMVRMM(NAME)                                    \
  static int lab_(NAME)(lua_State *L)                                   \
  {                                                                     \
    THTensor *r_ = NULL, *t = NULL, *mat = NULL, *vec = NULL;           \
    lua_Number beta = 1, alpha = 1;                                     \
    int narg = lua_gettop(L);                                           \
    int doacc = 0;                                                      \
                                                                        \
    if(narg > 0 && lua_isboolean(L, -1))                                \
    {                                                                   \
      doacc = lua_toboolean(L, -1);                                     \
      lua_pop(L, 1);                                                    \
      narg--;                                                           \
    }                                                                   \
                                                                        \
    if(narg == 6                                                        \
       && luaT_isudata(L, 1, torch_(Tensor_id))                         \
       && lua_isnumber(L, 2)                                            \
       && luaT_isudata(L, 3, torch_(Tensor_id))                         \
       && lua_isnumber(L, 4)                                            \
       && luaT_isudata(L, 5, torch_(Tensor_id))                         \
       && luaT_isudata(L, 6, torch_(Tensor_id)))                        \
    {                                                                   \
      r_ = luaT_toudata(L, 1, torch_(Tensor_id));                       \
      beta = lua_tonumber(L, 2);                                        \
      t = luaT_toudata(L, 3, torch_(Tensor_id));                        \
      alpha = lua_tonumber(L, 4);                                       \
      mat = luaT_toudata(L, 5, torch_(Tensor_id));                      \
      vec = luaT_toudata(L, 6, torch_(Tensor_id));                      \
    }                                                                   \
    else if(narg == 5                                                   \
            && luaT_isudata(L, 1, torch_(Tensor_id))                    \
            && lua_isnumber(L, 2)                                       \
            && luaT_isudata(L, 3, torch_(Tensor_id))                    \
            && luaT_isudata(L, 4, torch_(Tensor_id))                    \
            && luaT_isudata(L, 5, torch_(Tensor_id)))                   \
    {                                                                   \
      r_ = luaT_toudata(L, 1, torch_(Tensor_id));                       \
      beta = lua_tonumber(L, 2);                                        \
      t = luaT_toudata(L, 3, torch_(Tensor_id));                        \
      mat = luaT_toudata(L, 4, torch_(Tensor_id));                      \
      vec = luaT_toudata(L, 5, torch_(Tensor_id));                      \
    }                                                                   \
    else if(narg == 5                                                   \
            && luaT_isudata(L, 1, torch_(Tensor_id))                    \
            && luaT_isudata(L, 2, torch_(Tensor_id))                    \
            && lua_isnumber(L, 3)                                       \
            && luaT_isudata(L, 4, torch_(Tensor_id))                    \
            && luaT_isudata(L, 5, torch_(Tensor_id)))                   \
    {                                                                   \
      r_ = luaT_toudata(L, 1, torch_(Tensor_id));                       \
      t = luaT_toudata(L, 2, torch_(Tensor_id));                        \
      alpha = lua_tonumber(L, 3);                                       \
      mat = luaT_toudata(L, 4, torch_(Tensor_id));                      \
      vec = luaT_toudata(L, 5, torch_(Tensor_id));                      \
    }                                                                   \
    else if(narg == 5                                                   \
            && lua_isnumber(L, 1)                                       \
            && luaT_isudata(L, 2, torch_(Tensor_id))                    \
            && lua_isnumber(L, 3)                                       \
            && luaT_isudata(L, 4, torch_(Tensor_id))                    \
            && luaT_isudata(L, 5, torch_(Tensor_id)))                   \
    {                                                                   \
      beta = lua_tonumber(L, 1);                                        \
      t = luaT_toudata(L, 2, torch_(Tensor_id));                        \
      alpha = lua_tonumber(L, 3);                                       \
      mat = luaT_toudata(L, 4, torch_(Tensor_id));                      \
      vec = luaT_toudata(L, 5, torch_(Tensor_id));                      \
    }                                                                   \
    else if(narg == 4                                                   \
            && luaT_isudata(L, 1, torch_(Tensor_id))                    \
            && luaT_isudata(L, 2, torch_(Tensor_id))                    \
            && luaT_isudata(L, 3, torch_(Tensor_id))                    \
            && luaT_isudata(L, 4, torch_(Tensor_id)))                   \
    {                                                                   \
      r_ = luaT_toudata(L, 1, torch_(Tensor_id));                       \
      t = luaT_toudata(L, 2, torch_(Tensor_id));                        \
      mat = luaT_toudata(L, 3, torch_(Tensor_id));                      \
      vec = luaT_toudata(L, 4, torch_(Tensor_id));                      \
    }                                                                   \
    else if(narg == 4                                                   \
            && lua_isnumber(L, 1)                                       \
            && luaT_isudata(L, 2, torch_(Tensor_id))                    \
            && luaT_isudata(L, 3, torch_(Tensor_id))                    \
            && luaT_isudata(L, 4, torch_(Tensor_id)))                   \
    {                                                                   \
      beta = lua_tonumber(L, 1);                                        \
      t = luaT_toudata(L, 2, torch_(Tensor_id));                        \
      mat = luaT_toudata(L, 3, torch_(Tensor_id));                      \
      vec = luaT_toudata(L, 4, torch_(Tensor_id));                      \
    }                                                                   \
    else if(narg == 4                                                   \
            && luaT_isudata(L, 1, torch_(Tensor_id))                    \
            && lua_isnumber(L, 2)                                       \
            && luaT_isudata(L, 3, torch_(Tensor_id))                    \
            && luaT_isudata(L, 4, torch_(Tensor_id)))                   \
    {                                                                   \
      t = luaT_toudata(L, 1, torch_(Tensor_id));                        \
      alpha = lua_tonumber(L, 2);                                       \
      mat = luaT_toudata(L, 3, torch_(Tensor_id));                      \
      vec = luaT_toudata(L, 4, torch_(Tensor_id));                      \
    }                                                                   \
    else if(narg == 3                                                   \
            && luaT_isudata(L, 1, torch_(Tensor_id))                    \
            && luaT_isudata(L, 2, torch_(Tensor_id))                    \
            && luaT_isudata(L, 3, torch_(Tensor_id)))                   \
    {                                                                   \
      t = luaT_toudata(L, 1, torch_(Tensor_id));                        \
      mat = luaT_toudata(L, 2, torch_(Tensor_id));                      \
      vec = luaT_toudata(L, 3, torch_(Tensor_id));                      \
    }                                                                   \
    else                                                                \
      THError("invalid arguments: [result] [beta] tensor [alpha] tensor tensor [doacc]"); \
                                                                        \
    if(!r_)                                                             \
    {                                                                   \
      if(doacc)                                                         \
        r_ = t;                                                         \
      else                                                              \
        r_ = THTensor_(new)();                                          \
    }                                                                   \
    else                                                                \
      THTensor_(retain)(r_);                                            \
                                                                        \
    luaT_pushudata(L, r_, torch_(Tensor_id));                           \
                                                                        \
    THLab_(NAME)(r_, beta, t, alpha, mat, vec);                         \
                                                                        \
    return 1;                                                           \
  }

LAB_IMPLEMENT_ADDMVRMM(addmv)
LAB_IMPLEMENT_ADDMVRMM(addmm)
LAB_IMPLEMENT_ADDMVRMM(addr)

static int lab_(numel)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_(Tensor_id));
  lua_pushnumber(L, THLab_(numel)(tensor));
  return 1;
}

static int lab_(max_)(lua_State *L)
{
  THTensor *values_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THLongTensor *indices_ = luaT_checkudata(L, 2, torch_LongTensor_id);
  THTensor *t = luaT_checkudata(L, 3, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 4, THTensor_(nDimension)(t)))-1;

  THLab_(max)(values_, indices_, t, dimension);
  THLongLab_add(indices_, indices_, 1);

  lua_settop(L, 2);  
  return 2;
}

static int lab_(max)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 1 || ( n == 2 && lua_type(L,2) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
    luaT_pushudata(L, THLongTensor_new(), torch_LongTensor_id);
    lua_insert(L, 2);
  }
  return lab_(max_)(L);
}

static int lab_(min_)(lua_State *L)
{
  THTensor *values_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THLongTensor *indices_ = luaT_checkudata(L, 2, torch_LongTensor_id);
  THTensor *t = luaT_checkudata(L, 3, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 4, THTensor_(nDimension)(t)))-1;

  THLab_(min)(values_, indices_, t, dimension);
  THLongLab_add(indices_, indices_, 1);

  lua_settop(L, 2);  
  return 2;
}

static int lab_(min)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 1 || (n == 2 && lua_type(L,2) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
    luaT_pushudata(L, THLongTensor_new(), torch_LongTensor_id);
    lua_insert(L, 2);
  }
  return lab_(min_)(L);
}

static int lab_(sum_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 3, THTensor_(nDimension)(t)))-1;

  THLab_(sum)(r_, t, dimension);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(sum)(lua_State *L)
{  
  int n = lua_gettop(L);
  if (n == 1 || (n == 2 && lua_type(L,2) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(sum_)(L);
}

static int lab_(prod_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 3, THTensor_(nDimension)(t)))-1;

  THLab_(prod)(r_, t, dimension);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(prod)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 1 || (n == 2 && lua_type(L,2) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(prod_)(L);
}

static int lab_(cumsum_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 3, THTensor_(nDimension)(t)))-1;

  THLab_(cumsum)(r_, t, dimension);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(cumsum)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 1 || (n == 2 && lua_type(L,2) == LUA_TNUMBER)) 
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(cumsum_)(L);
}

static int lab_(cumprod_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 3, THTensor_(nDimension)(t)))-1;

  THLab_(cumprod)(r_, t, dimension);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(cumprod)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 1 || (n == 2 && lua_type(L,2) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(cumprod_)(L);
}

static int lab_(trace)(lua_State *L)
{
  THTensor *t = luaT_checkudata(L, 1, torch_(Tensor_id));
  lua_pushnumber(L, THLab_(trace)(t));
  return 1;
}

static int lab_(cross_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *a = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *b = luaT_checkudata(L, 3, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 4, -1))-1;

  THLab_(cross)(r_, a, b, dimension);

  lua_settop(L, 1);
  return 1;
}

static int lab_(cross)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 2 || (n == 3 && lua_type(L,3) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(cross_)(L);
}

static int lab_(zeros_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THLongStorage *dimension = lab_checklongargs(L, 2);

  THLab_(zeros)(r_, dimension);

  THLongStorage_free(dimension);
  lua_settop(L, 1);  
  return 1;
}

static int lab_(zeros)(lua_State *L)
{
  if (lua_type(L,1) == LUA_TNUMBER || ( lua_gettop(L) == 1 && luaT_isudata(L,1, torch_LongStorage_id)))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(zeros_)(L);
}

static int lab_(ones_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THLongStorage *dimension = lab_checklongargs(L, 2);

  THLab_(ones)(r_, dimension);

  THLongStorage_free(dimension);
  lua_settop(L, 1);  
  return 1;
}

static int lab_(ones)(lua_State *L)
{
  if (lua_type(L,1) == LUA_TNUMBER || ( lua_gettop(L) == 1 && luaT_isudata(L,1, torch_LongStorage_id)))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(ones_)(L);
}

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

static int lab_(tril_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int k = (int)(luaL_optnumber(L, 3, 0));

  THLab_(tril)(r_, t, k);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(tril)(lua_State *L)
{
  int n = lua_gettop(L);
  if ( n == 1 || (n == 2 && lua_type(L,2) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(tril_)(L);
}

static int lab_(triu_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int k = (int)(luaL_optnumber(L, 3, 0));

  THLab_(triu)(r_, t, k);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(triu)(lua_State *L)
{
  int n = lua_gettop(L);
  if ( n == 1 || (n == 2 && lua_type(L,2) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(triu_)(L);
}

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

static int lab_(convxcorr2)(lua_State *L,char* ktype)
{
  THTensor *r_ = NULL;
  THTensor *image = luaT_checkudata(L,1,torch_(Tensor_id));
  THTensor *kernel = luaT_checkudata(L,2,torch_(Tensor_id));
  int n = lua_gettop(L);
  const char* ctype = "v";
  if (n == 2)
  {
    r_ = THTensor_(new)();
    luaT_pushudata(L, r_, torch_(Tensor_id));
  }
  else if (n == 3)
  {
    if (luaT_isudata(L,3, torch_(Tensor_id)))
    {
      r_ = image;
      image = kernel;
      kernel = luaT_checkudata(L,3,torch_(Tensor_id));
      lua_settop(L,1);
    }
    else if (lua_isstring(L,3))
    {
      r_ = THTensor_(new)();
      ctype = luaL_checkstring(L,3);
      luaT_pushudata(L, r_, torch_(Tensor_id));
    }
    else
    {
      return luaL_error(L, "bad arguments: [result,] source, kernel [, conv type]");
    }
  }
  else if (n == 4)
  {
    r_ = image;
    image = kernel;
    kernel = luaT_checkudata(L,3,torch_(Tensor_id));
    ctype = luaL_checkstring(L,4);
    lua_settop(L,1);
  }
  else
  {
    return luaL_error(L, "bad arguments: [result,] source, kernel [, conv type]");
  }
  if (!r_)
  {
    return luaL_error(L, "oops, bad arguments: [result,] source, kernel [, conv type]");
  }
/*   else */
/*   { */
/*     //luaT_pushudata(L, r_, torch_(Tensor_id)); */
/*   } */

  char type[2];
  type[0] = ctype[0];
  type[1] = ktype[0];

  if (image->nDimension == 2 && kernel->nDimension == 2)
  {
    THLab_(conv2Dmul)(r_,0.0,1.0,image,kernel,1,1,type);
  }
  else if (image->nDimension == 3 && kernel->nDimension == 3)
  {
    THLab_(conv2Dger)(r_,0.0,1.0,image,kernel,1,1,type);
  }
  else if (image->nDimension == 3 && kernel->nDimension == 4)
  {
    THLab_(conv2Dmv)(r_,0.0,1.0,image,kernel,1,1,type);
  }
  else if (image->nDimension == 2 && kernel->nDimension == 3)
  {
    if (kernel->size[0] > 1)
    {
      long k;
      THTensor *ri = THTensor_(new)();
      THTensor *ker = THTensor_(new)();

      long nInputRows  = image->size[0];
      long nInputCols  = image->size[1];
      long nKernelRows = kernel->size[1];
      long nKernelCols = kernel->size[2];
      long nOutputRows, nOutputCols;

      THArgCheck((nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *type == 'f', 2, "Input image is smaller than kernel");
  
      if (type[0] == 'f') {
	nOutputRows = (nInputRows - 1) * 1 + nKernelRows;
	nOutputCols = (nInputCols - 1) * 1 + nKernelCols;
      } else { // valid
	nOutputRows = (nInputRows - nKernelRows) / 1 + 1;
	nOutputCols = (nInputCols - nKernelCols) / 1 + 1;
      }

      THTensor_(resize3d)(r_,kernel->size[0], nOutputRows, nOutputCols);
      for (k=0; k<kernel->size[0]; k++)
      {
        THTensor_(select)(ker,kernel,0,k);
        THTensor_(select)(ri,r_,0,k);
        THLab_(conv2Dmul)(ri,0.0,1.0,image,ker,1,1,type);
      }
      THTensor_(free)(ri);
      THTensor_(free)(ker);
    } else {
      THTensor *ker = THTensor_(new)();
      THTensor_(select)(ker,kernel,0,0);
      THLab_(conv2Dmul)(r_,0.0,1.0,image,ker,1,1,type);
      THTensor_(free)(ker);
    }
  }
  else if (image->nDimension == 3 && kernel->nDimension == 2)
  {
    if (image->size[0] > 1)
    {
      long k;
      THTensor *ri = THTensor_(new)();
      THTensor *im = THTensor_(new)();

      long nInputRows  = image->size[1];
      long nInputCols  = image->size[2];
      long nKernelRows = kernel->size[0];
      long nKernelCols = kernel->size[1];
      long nOutputRows, nOutputCols;

      THArgCheck((nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *type == 'f', 2, "Input image is smaller than kernel");
  
      if (type[0] == 'f') {
	nOutputRows = (nInputRows - 1) * 1 + nKernelRows;
	nOutputCols = (nInputCols - 1) * 1 + nKernelCols;
      } else { // valid
	nOutputRows = (nInputRows - nKernelRows) / 1 + 1;
	nOutputCols = (nInputCols - nKernelCols) / 1 + 1;
      }
      THTensor_(resize3d)(r_,image->size[0], nOutputRows, nOutputCols);
      for (k=0; k<image->size[0]; k++)
      {
        THTensor_(select)(im, image, 0, k);
        THTensor_(select)(ri,r_,0,k);
        THLab_(conv2Dmul)(ri,0.0,1.0,im,kernel,1,1,type);
      }
      THTensor_(free)(ri);
      THTensor_(free)(im);
    } else {
      THTensor *im = THTensor_(new)();
      THTensor_(select)(im,image,0,0);
      THLab_(conv2Dmul)(r_,0.0,1.0,im,kernel,1,1,type);
      THTensor_(free)(im);
    }
  }
  return 1;
}

static int lab_(conv2)(lua_State *L)
{
  return lab_(convxcorr2)(L,"convolution");
}
static int lab_(xcorr2)(lua_State *L)
{
  return lab_(convxcorr2)(L,"xcorrelation");
}

static int lab_(convxcorr3)(lua_State *L,char* ktype)
{
  THTensor *r_ = NULL;
  THTensor *image = luaT_checkudata(L,1,torch_(Tensor_id));
  THTensor *kernel = luaT_checkudata(L,2,torch_(Tensor_id));
  int n = lua_gettop(L);
  const char* ctype = "v";
  if (n == 2)
  {
    r_ = THTensor_(new)();
    luaT_pushudata(L, r_, torch_(Tensor_id));    
  }
  else if (n == 3)
  {
    if (luaT_isudata(L,3, torch_(Tensor_id)))
    {
      r_ = image;
      image = kernel;
      kernel = luaT_checkudata(L,3,torch_(Tensor_id));
      lua_settop(L,1);
    }
    else if (lua_isstring(L,3))
    {
      r_ = THTensor_(new)();
      ctype = luaL_checkstring(L,3);
      luaT_pushudata(L, r_, torch_(Tensor_id));
    }
    else
    {
      return luaL_error(L, "bad arguments: [result,] source, kernel [, conv type]");
    }
  }
  else if (n == 4)
  {
    r_ = image;
    image = kernel;
    kernel = luaT_checkudata(L,3,torch_(Tensor_id));
    ctype = luaL_checkstring(L,4);
    lua_settop(L,1);
  }
  else
  {
    return luaL_error(L, "bad arguments: [result,] source, kernel [, conv type]");
  }
  if (!r_)
  {
    return luaL_error(L, "oops, bad arguments: [result,] source, kernel [, conv type]");
  }
/*   else */
/*   { */
/*     luaT_pushudata(L, r_, torch_(Tensor_id)); */
/*   } */

  char type[2];
  type[0] = ctype[0];
  type[1] = ktype[0];

  if (image->nDimension == 3 && kernel->nDimension == 3)
  {
    THLab_(conv3Dmul)(r_,0.0,1.0,image,kernel,1,1,1,type);
  }
  else if (image->nDimension == 4 && kernel->nDimension == 4)
  {
    THLab_(conv3Dger)(r_,0.0,1.0,image,kernel,1,1,1,type);
  }
  else if (image->nDimension == 4 && kernel->nDimension == 5)
  {
    THLab_(conv3Dmv)(r_,0.0,1.0,image,kernel,1,1,1,type);
  }
  else if (image->nDimension == 3 && kernel->nDimension == 4)
  {
    if (kernel->size[0] > 1)
    {
      long k;
      THTensor *ri = THTensor_(new)();
      THTensor *ker = THTensor_(new)();

      long nInputDepth = image->size[0];
      long nInputRows  = image->size[1];
      long nInputCols  = image->size[2];
      long nKernelDepth= kernel->size[1];
      long nKernelRows = kernel->size[2];
      long nKernelCols = kernel->size[3];
      long nOutputDepth, nOutputRows, nOutputCols;

      THArgCheck((nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *type == 'f', 2, "Input image is smaller than kernel");
  
      if (type[0] == 'f') {
	nOutputDepth = (nInputDepth - 1) * 1 + nKernelDepth;
	nOutputRows = (nInputRows - 1) * 1 + nKernelRows;
	nOutputCols = (nInputCols - 1) * 1 + nKernelCols;
      } else { // valid
	nOutputDepth = (nInputDepth - nKernelDepth) / 1 + 1;
	nOutputRows = (nInputRows - nKernelRows) / 1 + 1;
	nOutputCols = (nInputCols - nKernelCols) / 1 + 1;
      }

      THTensor_(resize4d)(r_,kernel->size[0], nOutputDepth, nOutputRows, nOutputCols);
      for (k=0; k<kernel->size[0]; k++)
      {
        THTensor_(select)(ker,kernel,0,k);
        THTensor_(select)(ri,r_,0,k);
        THLab_(conv3Dmul)(ri,0.0,1.0,image,ker,1,1,1,type);
      }
      THTensor_(free)(ri);
      THTensor_(free)(ker);
    } else {
      THTensor *ker = THTensor_(new)();
      THTensor_(select)(ker,kernel,0,0);
      THLab_(conv3Dmul)(r_,0.0,1.0,image,ker,1,1,1,type);
      THTensor_(free)(ker);
    }
  }
  else if (image->nDimension == 4 && kernel->nDimension == 3)
  {
    if (image->size[0] > 1)
    {
      long k;
      THTensor *ri = THTensor_(new)();
      THTensor *im = THTensor_(new)();

      long nInputDepth = image->size[1];
      long nInputRows  = image->size[2];
      long nInputCols  = image->size[3];
      long nKernelDepth= kernel->size[0];
      long nKernelRows = kernel->size[1];
      long nKernelCols = kernel->size[2];
      long nOutputDepth, nOutputRows, nOutputCols;

      THArgCheck((nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *type == 'f', 2, "Input image is smaller than kernel");
  
      if (type[0] == 'f') {
	nOutputDepth = (nInputDepth - 1) * 1 + nKernelDepth;
	nOutputRows = (nInputRows - 1) * 1 + nKernelRows;
	nOutputCols = (nInputCols - 1) * 1 + nKernelCols;
      } else { // valid
	nOutputDepth = (nInputDepth - nKernelDepth) / 1 + 1;
	nOutputRows = (nInputRows - nKernelRows) / 1 + 1;
	nOutputCols = (nInputCols - nKernelCols) / 1 + 1;
      }
      THTensor_(resize4d)(r_,image->size[0], nOutputDepth, nOutputRows, nOutputCols);
      for (k=0; k<image->size[0]; k++)
      {
        THTensor_(select)(im, image, 0, k);
        THTensor_(select)(ri,r_,0,k);
        THLab_(conv3Dmul)(ri,0.0,1.0,im,kernel,1,1,1,type);
      }
      THTensor_(free)(ri);
      THTensor_(free)(im);
    } else {
      THTensor *im = THTensor_(new)();
      THTensor_(select)(im,image,0,0);
      THLab_(conv3Dmul)(r_,0.0,1.0,im,kernel,1,1,1,type);
      THTensor_(free)(im);
    }
  }
  return 1;
}

static int lab_(conv3)(lua_State *L)
{
  return lab_(convxcorr3)(L,"convolution");
}
static int lab_(xcorr3)(lua_State *L)
{
  return lab_(convxcorr3)(L,"xcorrelation");
}

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

static int lab_(gesv)(lua_State *L)
{
  THTensor *a_ = luaT_checkudata(L,1,torch_(Tensor_id));
  THTensor *b_ = luaT_checkudata(L,2,torch_(Tensor_id));
  int n = lua_gettop(L);
  if (n == 2 || (n == 3 && luaT_optboolean(L,3,1)))
  {
    // we want new stuff
    THTensor *ta = THTensor_(newClone)(a_);
    THTensor *tb = THTensor_(newClone)(b_);
    THLab_(gesv)(ta,tb);
    // clean ta
    THTensor_(free)(ta);
    // return tb
    luaT_pushudata(L, tb, torch_(Tensor_id));
    lua_insert(L,1);
    lua_settop(L,1);
  }
  else if (n == 3)
  {
    // just run like this
    THLab_(gesv)(a_,b_);
    lua_settop(L,2);
  }
  else if (n == 4)
  {
    THTensor *ta = luaT_checkudata(L,3,torch_(Tensor_id));
    THTensor *tb = luaT_checkudata(L,4,torch_(Tensor_id));
    THTensor_(resizeAs)(b_,tb);
    THTensor_(resizeAs)(a_,ta);
    THTensor_(copy)(b_,tb);
    THTensor_(copy)(a_,ta);
    THLab_(gesv)(a_,b_);
    // do not free anything, because user passed everything
    lua_settop(L,2);
  }
  else
  {
    luaL_error(L, " bad arguments: [TA,TB,] a,b or a,b [,flag] ");
  }
  return 1;
}

static int lab_(gels)(lua_State *L)
{
  THTensor *a_ = luaT_checkudata(L,1,torch_(Tensor_id));
  THTensor *b_ = luaT_checkudata(L,2,torch_(Tensor_id));
  int n = lua_gettop(L);
  if (n == 2 || (n == 3 && luaT_optboolean(L,3,1)))
  {
    // we want new stuff
    THTensor *ta = THTensor_(newClone)(a_);
    THTensor *tb = THTensor_(newClone)(b_);
    THLab_(gels)(ta,tb);
    // clean ta
    THTensor_(free)(ta);
    // return tb
    luaT_pushudata(L, tb, torch_(Tensor_id));
    lua_insert(L,1);
    lua_settop(L,1);
  }
  else if (n == 3)
  {
    // just run like this
    THLab_(gels)(a_,b_);
    lua_settop(L,2);
  }
  else if (n == 4)
  {
    THTensor *ta = luaT_checkudata(L,3,torch_(Tensor_id));
    THTensor *tb = luaT_checkudata(L,4,torch_(Tensor_id));
    THTensor_(resizeAs)(b_,tb);
    THTensor_(resizeAs)(a_,ta);
    THTensor_(copy)(b_,tb);
    THTensor_(copy)(a_,ta);
    THLab_(gels)(a_,b_);
    // do not free anything, because user passed everything
    lua_settop(L,2);
  }
  else
  {
    luaL_error(L, " bad arguments: [TA,TB,] a,b or a,b [,flag] ");
  }
  return 1;
}

static int lab_(eig)(lua_State *L)
{
  THTensor *a_, *e_;

  //e=(a), e,v=(a,'v'), e=(e,a), e,v=(e,v,a)
  int n = lua_gettop(L);
  if (n == 1)
  {
    THTensor *ta = luaT_checkudata(L,1,torch_(Tensor_id));
    a_ = THTensor_(newClone)(ta);
    e_ = THTensor_(new)();
    luaT_pushudata(L, e_, torch_(Tensor_id));
    lua_insert(L,1);
    lua_settop(L,1);
    THLab_(syev)(a_,e_,"N","U");
    THTensor_(free)(a_);
    return 1;
  }
  else if (n == 2 && lua_type(L,2) != LUA_TSTRING)//e=(e,a)
  {
    e_ = luaT_checkudata(L,1,torch_(Tensor_id));
    THTensor *ta = luaT_checkudata(L,2,torch_(Tensor_id));
    a_ = THTensor_(newClone)(ta);
    lua_settop(L,1);
    THLab_(syev)(a_,e_,"N","U");
    THTensor_(free)(a_);
    return 1;
  }
  else if (n == 2 && lua_type(L,2) == LUA_TSTRING)//e,v=(a,'v')
  {
    const char *type = luaL_checkstring(L,2);
    luaL_argcheck(L, (type[0] == 'v' || type[0] == 'V' || type[0] == 'n' || type[0] == 'N'),
		  2, "expected 'n' or 'v' for (eigenvals or vals+vectors)");
    if (type[0] == 'v' || type[0] == 'V')
    {
      THTensor *ta = luaT_checkudata(L,1,torch_(Tensor_id));
      a_ = THTensor_(newClone)(ta);
      e_ = THTensor_(new)();
      luaT_pushudata(L, e_, torch_(Tensor_id));
      lua_insert(L,1);
      luaT_pushudata(L, a_, torch_(Tensor_id));
      lua_insert(L,2);
      lua_settop(L,2);
      THLab_(syev)(a_,e_,"V","U");
      return 2;
    }
    else
    {
      THTensor *ta = luaT_checkudata(L,1,torch_(Tensor_id));
      a_ = THTensor_(newClone)(ta);
      e_ = THTensor_(new)();
      luaT_pushudata(L, e_, torch_(Tensor_id));
      lua_insert(L,1);
      lua_settop(L,1);
      THLab_(syev)(a_,e_,"N","U");
      THTensor_(free)(a_);
      return 1;
    }
  }
  else if (n == 3)//e,v=(e,v,a)
  {
    e_ = luaT_checkudata(L,1,torch_(Tensor_id));
    a_ = luaT_checkudata(L,2,torch_(Tensor_id));
    THTensor *ta = luaT_checkudata(L,3,torch_(Tensor_id));
    THTensor_(resizeAs)(a_,ta);
    THTensor_(copy)(a_,ta);
    lua_settop(L,2);
    THLab_(syev)(a_,e_,"V","U");
    return 2;
  }
  else
  {
    luaL_error(L, " bad arguments: [e,v,] a");
  }
  return 0;
}

static int lab_(svd)(lua_State *L)
{
  THTensor *a_, *u_, *s_, *vt_;

  //u,s,v=(a), u,s,v=(a,'A'), u,s,v=(a,'S') 
  //u,s,v=(u,s,v,a), u,s,v=(u,s,v,a,'A'), u,s,v=(u,s,v,a,'S')
  int n = lua_gettop(L);
  char type = 'S';
  if (lua_type(L,n) == LUA_TSTRING)
  {
    const char *tt = luaL_checkstring(L,2);
    type = *tt;
    //printf("type= %c\n",type);
    luaL_argcheck(L, (type == 'a' || type == 'A' || type == 's' || type == 'S'),
		  n, "expected 'a' or 's' for (All or Some)");
    if (type == 'a') type = 'A';
    if (type == 's') type = 'S';
  }
  //printf("type = %c\n",type);
  if (n == 1 || n == 2)
  {
    THTensor *ta = luaT_checkudata(L,1,torch_(Tensor_id));
    a_ = THTensor_(newClone)(ta);
    u_ = THTensor_(new)();
    luaT_pushudata(L, u_, torch_(Tensor_id));
    lua_insert(L,1);
    s_ = THTensor_(new)();
    luaT_pushudata(L, s_, torch_(Tensor_id));
    lua_insert(L,2);
    vt_ = THTensor_(new)();
    luaT_pushudata(L, vt_, torch_(Tensor_id));
    lua_insert(L,3);
    lua_settop(L,3);

    THLab_(gesvd)(a_,s_,u_,vt_,type);
    THTensor_(free)(a_);
    return 3;
  }
  else if (n == 4 || n == 5)//u,s,v=(u,s,v,a)
  {
    u_ = luaT_checkudata(L,1,torch_(Tensor_id));
    s_ = luaT_checkudata(L,2,torch_(Tensor_id));
    vt_ = luaT_checkudata(L,3,torch_(Tensor_id));
    THTensor *ta = luaT_checkudata(L,4,torch_(Tensor_id));
    a_ = THTensor_(newClone)(ta);
    lua_settop(L,3);

    THLab_(gesvd)(a_,s_,u_,vt_,type);
    THTensor_(free)(a_);
    return 3;
  }
  else
  {
    luaL_error(L, " bad arguments: [u,s,v,] a ,[type]");
  }
  return 0;
}

static int lab_(mean_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = (int)(luaL_optnumber(L, 3, THTensor_(nDimension)(t)))-1;

  THLab_(mean)(r_, t, dimension);

  lua_settop(L, 1);  
  return 1;
}

static int lab_(mean)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 1 || (n == 2 && lua_type(L,2) == LUA_TNUMBER))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(mean_)(L);
}

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

static int lab_(meanall)(lua_State *L)
{
  THTensor *t = luaT_checkudata(L, 1, torch_(Tensor_id));  
  lua_pushnumber(L, THLab_(meanall)(t));
  return 1;
}

static int lab_(varall)(lua_State *L)
{
  THTensor *t = luaT_checkudata(L, 1, torch_(Tensor_id));  
  lua_pushnumber(L, THLab_(varall)(t));
  return 1;
}

static int lab_(stdall)(lua_State *L)
{
  THTensor *t = luaT_checkudata(L, 1, torch_(Tensor_id));  
  lua_pushnumber(L, THLab_(stdall)(t));
  return 1;
}

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

static int lab_(rand_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THLongStorage *dimension = lab_checklongargs(L, 2);

  THLab_(rand)(r_, dimension);

  THLongStorage_free(dimension);
  lua_settop(L, 1);  
  return 1;
}

static int lab_(rand)(lua_State *L)
{
  if (lua_type(L,1) == LUA_TNUMBER || ( lua_gettop(L) == 1 && luaT_isudata(L,1, torch_LongStorage_id)))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(rand_)(L);
}

static int lab_(randn_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THLongStorage *dimension = lab_checklongargs(L, 2);

  THLab_(randn)(r_, dimension);

  THLongStorage_free(dimension);
  lua_settop(L, 1);  
  return 1;
}

static int lab_(randn)(lua_State *L)
{
  if (lua_type(L,1) == LUA_TNUMBER || ( lua_gettop(L) == 1 && luaT_isudata(L,1, torch_LongStorage_id)))
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  return lab_(randn_)(L);
}

#define LAB_IMPLEMENT_MATH_FUNC(NAME)                         \
  static int lab_(NAME##_)(lua_State *L)                      \
  {                                                           \
    THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));  \
    THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));   \
                                                              \
    THLab_(NAME)(r_, t);                                      \
                                                              \
    lua_settop(L, 1);                                         \
    return 1;                                                 \
  }                                                           \
                                                              \
  static int lab_(NAME)(lua_State *L)                         \
  {                                                           \
    int n = lua_gettop(L);				      \
    if(n == 1)						      \
    {							      \
      if (lua_isnumber(L, 1))                                 \
        {                                                     \
          real in = lua_tonumber(L, 1);                       \
          real out = NAME(in);                                \
          lua_pushnumber(L, out);                             \
          return 1;                                           \
        }                                                     \
      luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id)); \
      lua_insert(L, 1);					      \
    }							      \
    else if (n != 2)					      \
    {							      \
      luaL_error(L, "bad arguments: [tensor, ] tensor");      \
    }							      \
    return lab_(NAME##_)(L);                                  \
  }                                                           \
                                                              \

LAB_IMPLEMENT_MATH_FUNC(log)
LAB_IMPLEMENT_MATH_FUNC(log1p)
LAB_IMPLEMENT_MATH_FUNC(exp)
LAB_IMPLEMENT_MATH_FUNC(cos)
LAB_IMPLEMENT_MATH_FUNC(acos)
LAB_IMPLEMENT_MATH_FUNC(cosh)
LAB_IMPLEMENT_MATH_FUNC(sin)
LAB_IMPLEMENT_MATH_FUNC(asin)
LAB_IMPLEMENT_MATH_FUNC(sinh)
LAB_IMPLEMENT_MATH_FUNC(tan)
LAB_IMPLEMENT_MATH_FUNC(atan)
LAB_IMPLEMENT_MATH_FUNC(tanh)
LAB_IMPLEMENT_MATH_FUNC(sqrt)
LAB_IMPLEMENT_MATH_FUNC(ceil)
LAB_IMPLEMENT_MATH_FUNC(floor)
LAB_IMPLEMENT_MATH_FUNC(abs)

static int lab_(pow_)(lua_State *L)
{
  THTensor *r_ = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *t = luaT_checkudata(L, 2, torch_(Tensor_id));
  real value = (real)luaL_checknumber(L, 3);
  THLab_(pow)(r_, t, value);
  lua_settop(L, 1);
  return 1;
}

static int lab_(pow)(lua_State *L)
{
  int n = lua_gettop(L);
  if (n == 2 )
  {
    luaT_pushudata(L, THTensor_(new)(), torch_(Tensor_id));
    lua_insert(L, 1);
  }
  else if (n != 3)
  {
    luaL_error(L, "bad arguments: [tensor, ] tensor, power");
  }
  return lab_(pow_)(L);
}

#endif

static const struct luaL_Reg lab_(stuff__) [] = {
  {"numel", lab_(numel)},
  //{"max_", lab_(max_)},
  {"max", lab_(max)},
  //{"min_", lab_(min_)},
  {"min", lab_(min)},
  //{"sum_", lab_(sum_)},
  {"sum", lab_(sum)},
  //{"prod_", lab_(prod_)},
  {"prod", lab_(prod)},
  //{"cumsum_", lab_(cumsum_)},
  {"cumsum", lab_(cumsum)},
  //{"cumprod_", lab_(cumprod_)},
  {"cumprod", lab_(cumprod)},
  {"trace", lab_(trace)},
  //{"cross_", lab_(cross_)},
  {"cross", lab_(cross)},
  {"zeros_", lab_(zeros_)},
  {"zeros", lab_(zeros)},
  {"ones_", lab_(ones_)},
  {"ones", lab_(ones)},
  //{"diag_", lab_(diag_)},
  {"diag", lab_(diag)},
  //{"eye_", lab_(eye_)},
  {"eye", lab_(eye)},
  //{"range_", lab_(range_)},
  {"range", lab_(range)},
  //{"randperm_", lab_(randperm_)},
  {"randperm", lab_(randperm)},
  {"reshape_", lab_(reshape_)},
  {"reshape", lab_(reshape)},
  //{"sort_", lab_(sort_)},
  {"sort", lab_(sort)},
  //{"tril_", lab_(tril_)},
  {"tril", lab_(tril)},
  //{"triu_", lab_(triu_)},
  {"triu", lab_(triu)},
  {"_histc", lab_(histc)},
  //{"cat_", lab_(cat_)},
  {"cat", lab_(cat)},
  {"conv2", lab_(conv2)},
  {"xcorr2", lab_(xcorr2)},
  {"conv3", lab_(conv3)},
  {"xcorr3", lab_(xcorr3)},
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  //#ifdef __LAPACK__
  {"gesv", lab_(gesv)},
  {"gels", lab_(gels)},
  {"eig", lab_(eig)},
  {"svd", lab_(svd)},
  //#endif
  //{"log_", lab_(log_)},
  {"log", lab_(log)},
  //{"log1p_", lab_(log1p_)},
  {"log1p", lab_(log1p)},
  //{"exp_", lab_(exp_)},
  {"exp", lab_(exp)},
  //{"cos_", lab_(cos_)},
  {"cos", lab_(cos)},
  //{"acos_", lab_(acos_)},
  {"acos", lab_(acos)},
  //{"cosh_", lab_(cosh_)},
  {"cosh", lab_(cosh)},
  //{"sin_", lab_(sin_)},
  {"sin", lab_(sin)},
  //{"asin_", lab_(asin_)},
  {"asin", lab_(asin)},
  //{"sinh_", lab_(sinh_)},
  {"sinh", lab_(sinh)},
  //{"tan_", lab_(tan_)},
  {"tan", lab_(tan)},
  //{"atan_", lab_(atan_)},
  {"atan", lab_(atan)},
  //{"tanh_", lab_(tanh_)},
  {"tanh", lab_(tanh)},
  //{"pow_", lab_(pow_)},
  {"pow", lab_(pow)},
  //{"sqrt_", lab_(sqrt_)},
  {"sqrt", lab_(sqrt)},
  //{"ceil_", lab_(ceil_)},
  {"ceil", lab_(ceil)},
  //{"floor_", lab_(floor_)},
  {"floor", lab_(floor)},
  //{"abs_", lab_(abs_)},
  {"abs", lab_(abs)},
  //{"mean_", lab_(mean_)},
  {"mean", lab_(mean)},
  //{"std_", lab_(std_)},
  {"std", lab_(std)},
  //{"var_", lab_(var_)},
  {"var", lab_(var)},
  {"norm", lab_(norm)},
  {"dist", lab_(dist)},
  {"meanall", lab_(meanall)},
  {"varall", lab_(varall)},
  {"stdall", lab_(stdall)},
  //{"linspace_", lab_(linspace_)},
  {"linspace", lab_(linspace)},
  //{"logspace_", lab_(logspace_)},
  {"logspace", lab_(logspace)},
  {"rand_", lab_(rand_)},
  {"rand", lab_(rand)},
  {"randn_", lab_(randn_)},
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
