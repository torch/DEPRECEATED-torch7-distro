#define LAB_IMPLEMENT_T(NAME)                                       \
  static int lab_(NAME)(lua_State *L)                               \
  {                                                                 \
    THTensor *tensor = NULL;                                        \
                                                                    \
    if(lua_gettop(L) == 1 && luaT_isudata(L, 1, torch_(Tensor_id))) \
    {                                                               \
      tensor = luaT_toudata(L, 1, torch_(Tensor_id));               \
      THLab_(NAME)(tensor);                                         \
    }                                                               \
    else                                                            \
      luaL_error(L, "invalid arguments: tensor");                   \
                                                                    \
    return 1;                                                       \
  }

#define LAB_IMPLEMENT_rNT(NAME)                                     \
  static int lab_(NAME)(lua_State *L)                               \
  {                                                                 \
    THTensor *tensor = NULL;                                        \
                                                                    \
    if(lua_gettop(L) == 1 && luaT_isudata(L, 1, torch_(Tensor_id))) \
    {                                                               \
      tensor = luaT_toudata(L, 1, torch_(Tensor_id));               \
      lua_pushnumber(L, THLab_(NAME)(tensor));                      \
    }                                                               \
    else                                                            \
      luaL_error(L, "invalid arguments: tensor");                   \
                                                                    \
    return 1;                                                       \
  }

#define LAB_IMPLEMENT_oTTRoA(NAME)                                \
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
      luaL_error(L, "invalid arguments: [tensor] tensor number [boolean]"); \
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

#define LAB_IMPLEMENT_oTTToA(NAME)                          \
  static int lab_(NAME)(lua_State *L)                       \
  {                                                         \
    THTensor *r_ = NULL, *t = NULL, *src = NULL;            \
    int narg = lua_gettop(L);                               \
    int doacc = 0;                                          \
                                                            \
    if(narg > 0 && lua_isboolean(L, -1))                    \
    {                                                       \
      doacc = lua_toboolean(L, -1);                         \
      lua_pop(L, 1);                                        \
      narg--;                                               \
    }                                                       \
                                                            \
    if(narg == 3                                            \
       && luaT_isudata(L, 1, torch_(Tensor_id))             \
       && luaT_isudata(L, 2, torch_(Tensor_id))             \
       && luaT_isudata(L, 3, torch_(Tensor_id)))            \
    {                                                       \
      r_ = luaT_toudata(L, 1, torch_(Tensor_id));           \
      t = luaT_toudata(L, 2, torch_(Tensor_id));            \
      src = luaT_toudata(L, 3, torch_(Tensor_id));          \
    }                                                       \
    else if(narg == 3                                       \
            && luaT_isudata(L, 1, torch_(Tensor_id))        \
            && luaT_isudata(L, 2, torch_(Tensor_id)))       \
    {                                                       \
      t = luaT_toudata(L, 1, torch_(Tensor_id));            \
      src = luaT_toudata(L, 2, torch_(Tensor_id));          \
    }                                                       \
    else                                                    \
      THError("invalid arguments: [result] tensor tensor"); \
                                                            \
    if(!r_)                                                 \
    {                                                       \
      if(doacc)                                             \
        r_ = t;                                             \
      else                                                  \
        r_ = THTensor_(new)();                              \
    }                                                       \
    else                                                    \
      THTensor_(retain)(r_);                                \
                                                            \
    luaT_pushudata(L, r_, torch_(Tensor_id));               \
                                                            \
    THLab_(NAME)(r_, t, src);                               \
                                                            \
    return 1;                                               \
  }

#define LAB_IMPLEMENT_oTToRTToA(NAME)                                   \
  static int lab_(NAME)(lua_State *L)                                   \
  {                                                                     \
    THTensor *r_ = NULL, *t = NULL, *src1 = NULL, *src2 = NULL;         \
    real value = 1;                                                     \
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
    if(narg == 5                                                        \
       && luaT_isudata(L, 1, torch_(Tensor_id))                         \
       && luaT_isudata(L, 2, torch_(Tensor_id))                         \
       && lua_isnumber(L, 3)                                            \
       && luaT_isudata(L, 4, torch_(Tensor_id))                         \
       && luaT_isudata(L, 5, torch_(Tensor_id)))                        \
    {                                                                   \
      r_ = luaT_toudata(L, 1, torch_(Tensor_id));                       \
      t = luaT_toudata(L, 2, torch_(Tensor_id));                        \
      value = lua_tonumber(L, 3);                                       \
      src1 = luaT_toudata(L, 4, torch_(Tensor_id));                     \
      src2 = luaT_toudata(L, 5, torch_(Tensor_id));                     \
    }                                                                   \
    else if(narg == 4                                                   \
            && luaT_isudata(L, 1, torch_(Tensor_id))                    \
            && luaT_isudata(L, 2, torch_(Tensor_id))                    \
            && luaT_isudata(L, 3, torch_(Tensor_id))                    \
            && luaT_isudata(L, 4, torch_(Tensor_id)))                   \
    {                                                                   \
      r_ = luaT_toudata(L, 1, torch_(Tensor_id));                       \
      t = luaT_toudata(L, 2, torch_(Tensor_id));                        \
      src1 = luaT_toudata(L, 3, torch_(Tensor_id));                     \
      src2 = luaT_toudata(L, 4, torch_(Tensor_id));                     \
    }                                                                   \
    else if(narg == 4                                                   \
            && luaT_isudata(L, 1, torch_(Tensor_id))                    \
            && lua_isnumber(L, 2)                                       \
            && luaT_isudata(L, 3, torch_(Tensor_id))                    \
            && luaT_isudata(L, 4, torch_(Tensor_id)))                   \
    {                                                                   \
      t = luaT_toudata(L, 1, torch_(Tensor_id));                        \
      value = lua_tonumber(L, 2);                                       \
      src1 = luaT_toudata(L, 3, torch_(Tensor_id));                     \
      src2 = luaT_toudata(L, 4, torch_(Tensor_id));                     \
    }                                                                   \
    else                                                                \
      THError("invalid arguments: [result] tensor [number] tensor tensor"); \
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
    THLab_(NAME)(r_, t, value, src1, src2);                             \
                                                                        \
    return 1;                                                           \
  }

#define LAB_IMPLEMENT_oToRToRTToA(NAME)                                 \
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

#define LAB_IMPLEMENT_oTToI(NAME)                                       \
  static int lab_(NAME)(lua_State *L)                                   \
  {                                                                     \
    THTensor *r_ = NULL, *t = NULL;                                     \
    int dimension = 0;                                                  \
    int narg = lua_gettop(L);                                           \
                                                                        \
    if(narg == 1                                                        \
       && luaT_checkudata(L, 1, torch_(Tensor_id)))                     \
    {                                                                   \
      t = luaT_toudata(L, 1, torch_(Tensor_id));                        \
    }                                                                   \
    else if(narg == 2                                                   \
            && luaT_checkudata(L, 1, torch_(Tensor_id))                 \
            && luaT_checkudata(L, 2, torch_(Tensor_id)))                \
    {                                                                   \
      r_ = luaT_toudata(L, 1, torch_(Tensor_id));                       \
      t = luaT_toudata(L, 2, torch_(Tensor_id));                        \
    }                                                                   \
    else if(narg == 2                                                   \
            && luaT_checkudata(L, 1, torch_(Tensor_id))                 \
            && lua_isnumber(L, 2))                                      \
    {                                                                   \
      t = luaT_toudata(L, 1, torch_(Tensor_id));                        \
      dimension = lua_tonumber(L, 2)-1;                                 \
    }                                                                   \
    else if(narg == 3                                                   \
            && luaT_checkudata(L, 1, torch_(Tensor_id))                 \
            && luaT_checkudata(L, 2, torch_(Tensor_id))                 \
            && lua_isnumber(L, 3))                                      \
    {                                                                   \
      r_ = luaT_toudata(L, 1, torch_(Tensor_id));                       \
      t = luaT_toudata(L, 2, torch_(Tensor_id));                        \
      dimension = lua_tonumber(L, 3)-1;                                 \
    }                                                                   \
    else                                                                \
      luaL_error(L, "invalid arguments: [result] tensor [integer]");    \
                                                                        \
    if(!r_)                                                             \
      r_ = THTensor_(new)();                                            \
    else                                                                \
      THTensor_(retain)(r_);                                            \
    luaT_pushudata(L, r_, torch_(Tensor_id));                           \
                                                                        \
    THLab_(NAME)(r_, t, dimension);                                     \
                                                                        \
    return 1;                                                           \
  }

#define LAB_IMPLEMENT_oTL(NAME)                                         \
  static int lab_(NAME)(lua_State *L)                                   \
  {                                                                     \
    THTensor *r_ = NULL;                                                \
    THLongStorage *dimension = NULL;                                    \
    int narg = lua_gettop(L);                                           \
                                                                        \
    if(narg >= 2                                                        \
       && luaT_isudata(L, 1, torch_(Tensor_id))                         \
       && lab_islongargs(L, 2))                                         \
    {                                                                   \
      r_ = luaT_toudata(L, 1, torch_(Tensor_id));                       \
      dimension = lab_checklongargs(L, 2);                              \
    }                                                                   \
    else if(narg >= 1                                                   \
            && lab_islongargs(L, 2))                                    \
    {                                                                   \
      dimension = lab_checklongargs(L, 2);                              \
    }                                                                   \
    else                                                                \
      luaL_error(L, "invalid arguments: [tensor] (storage | dim1 [dim2...])"); \
                                                                        \
    if(r_)                                                              \
      THTensor_(retain)(r_);                                            \
    else                                                                \
      r_ = THTensor_(new)();                                            \
    luaT_pushudata(L, r_, torch_(Tensor_id));                           \
                                                                        \
    THLab_(NAME)(r_, dimension);                                        \
                                                                        \
    THLongStorage_free(dimension);                                      \
                                                                        \
    return 1;                                                           \
  }
