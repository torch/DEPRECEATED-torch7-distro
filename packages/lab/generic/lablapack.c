#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/lablapack.c"
#else

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
    THTensor_(gesv)(ta,tb);
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
    THTensor_(gesv)(a_,b_);
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
    THTensor_(gesv)(a_,b_);
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
    THTensor_(gels)(ta,tb);
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
    THTensor_(gels)(a_,b_);
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
    THTensor_(gels)(a_,b_);
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
    THTensor_(syev)(a_,e_,"N","U");
    THTensor_(free)(a_);
    return 1;
  }
  else if (n == 2 && lua_type(L,2) != LUA_TSTRING)//e=(e,a)
  {
    e_ = luaT_checkudata(L,1,torch_(Tensor_id));
    THTensor *ta = luaT_checkudata(L,2,torch_(Tensor_id));
    a_ = THTensor_(newClone)(ta);
    lua_settop(L,1);
    THTensor_(syev)(a_,e_,"N","U");
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
      THTensor_(syev)(a_,e_,"V","U");
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
      THTensor_(syev)(a_,e_,"N","U");
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
    THTensor_(syev)(a_,e_,"V","U");
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

    THTensor_(gesvd)(a_,s_,u_,vt_,type);
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

    THTensor_(gesvd)(a_,s_,u_,vt_,type);
    THTensor_(free)(a_);
    return 3;
  }
  else
  {
    luaL_error(L, " bad arguments: [u,s,v,] a ,[type]");
  }
  return 0;
}

static const struct luaL_Reg lab_(lapack_stuff__) [] = {
  {"gesv", lab_(gesv)},
  {"gels", lab_(gels)},
  {"eig", lab_(eig)},
  {"svd", lab_(svd)},
  {NULL, NULL}
};

void lab_(lapack_init)(lua_State *L)
{
  torch_(Tensor_id) = luaT_checktypename2id(L, torch_string_(Tensor));

  /* register everything into the "lab" field of the tensor metaclass */
  luaT_pushmetaclass(L, torch_(Tensor_id));
  lua_pushstring(L, "lab");
  lua_rawget(L, -2);
  luaL_register(L, NULL, lab_(lapack_stuff__));
  lua_pop(L, 2);
}

#endif
