/* Storage type */
#define STORAGE_T_(TYPE) TH##TYPE##Storage
#define STORAGE_T(TYPE) STORAGE_T_(TYPE)
#define STORAGE STORAGE_T(CAP_TYPE)

/* Function name for a Storage */
#define STORAGE_FUNC_TN_(TYPE,NAME) TH##TYPE##Storage_##NAME
#define STORAGE_FUNC_TN(TYPE, NAME) STORAGE_FUNC_TN_(TYPE,NAME) 
#define STORAGE_FUNC(NAME) STORAGE_FUNC_TN(CAP_TYPE, NAME)

/* Wrapper function name for a Storage */
#define W_STORAGE_FUNC_TN_(TYPE, NAME) torch_##TYPE##Storage_##NAME
#define W_STORAGE_FUNC_TN(TYPE, NAME) W_STORAGE_FUNC_TN_(TYPE, NAME)
#define W_STORAGE_FUNC(NAME) W_STORAGE_FUNC_TN(CAP_TYPE, NAME)

/* Name and id in Lua */
#define LUA_STORAGE_NAME_T_(TYPE) "torch." #TYPE "Storage"
#define LUA_STORAGE_NAME_T(TYPE) LUA_STORAGE_NAME_T_(TYPE)
#define LUA_STORAGE W_STORAGE_FUNC(id)

static int W_STORAGE_FUNC(new)(lua_State *L)
{
  STORAGE *storage;
  if(lua_type(L, 1) == LUA_TSTRING)
  {
    const char *fileName = luaL_checkstring(L, 1);
    int isShared = luaT_optboolean(L, 2, 0);
    storage = STORAGE_FUNC(newWithMapping)(fileName, isShared);  }
  else
  {
    long size = luaL_optlong(L, 1, 0);
    storage = STORAGE_FUNC(newWithSize)(size);
  }
  luaT_pushudata(L, storage, LUA_STORAGE);
  return 1;
}

static int W_STORAGE_FUNC(free)(lua_State *L)
{
  STORAGE *storage = luaT_checkudata(L, 1, LUA_STORAGE);
  STORAGE_FUNC(free)(storage);
  return 0;
}

static int W_STORAGE_FUNC(resize)(lua_State *L)
{
  STORAGE *storage = luaT_checkudata(L, 1, LUA_STORAGE);
  long size = luaL_checklong(L, 2);
  int keepContent = luaT_optboolean(L, 3, 0);
  STORAGE_FUNC(resize)(storage, size, keepContent);
  lua_settop(L, 1);
  return 1;
}

static int W_STORAGE_FUNC(copy)(lua_State *L)
{
  STORAGE *storage = luaT_checkudata(L, 1, LUA_STORAGE);
  void *src;
  if( (src = luaT_toudata(L, 2, LUA_STORAGE)) )
    STORAGE_FUNC(copy)(storage, src);
  else if( (src = luaT_toudata(L, 2, torch_ByteStorage_id)) )
    STORAGE_FUNC(copyByte)(storage, src);
  else if( (src = luaT_toudata(L, 2, torch_CharStorage_id)) )
    STORAGE_FUNC(copyChar)(storage, src);
  else if( (src = luaT_toudata(L, 2, torch_ShortStorage_id)) )
    STORAGE_FUNC(copyShort)(storage, src);
  else if( (src = luaT_toudata(L, 2, torch_IntStorage_id)) )
    STORAGE_FUNC(copyInt)(storage, src);
  else if( (src = luaT_toudata(L, 2, torch_LongStorage_id)) )
    STORAGE_FUNC(copyLong)(storage, src);
  else if( (src = luaT_toudata(L, 2, torch_FloatStorage_id)) )
    STORAGE_FUNC(copyFloat)(storage, src);
  else if( (src = luaT_toudata(L, 2, torch_DoubleStorage_id)) )
    STORAGE_FUNC(copyDouble)(storage, src);
  else
    luaL_typerror(L, 2, "torch.*Storage");
  lua_settop(L, 1);
  return 1;
}

static int W_STORAGE_FUNC(fill)(lua_State *L)
{
  STORAGE *storage = luaT_checkudata(L, 1, LUA_STORAGE);
  double value = luaL_checknumber(L, 2);
  STORAGE_FUNC(fill)(storage, (TYPE)value);
  lua_settop(L, 1);
  return 1;
}

static int W_STORAGE_FUNC(__len__)(lua_State *L)
{
  STORAGE *storage = luaT_checkudata(L, 1, LUA_STORAGE);
  lua_pushnumber(L, storage->size);
  return 1;
}

static int W_STORAGE_FUNC(__newindex__)(lua_State *L)
{
  if(lua_isnumber(L, 2))
  {
    STORAGE *storage = luaT_checkudata(L, 1, LUA_STORAGE);
    long index = luaL_checklong(L, 2) - 1;
    double number = luaL_checknumber(L, 3);
    luaL_argcheck(L, 0 <= index && index < storage->size, 2, "index out of range");
    storage->data[index] = (TYPE)number;
    lua_pushboolean(L, 1);
  }
  else
    lua_pushboolean(L, 0);

  return 1;
}

static int W_STORAGE_FUNC(__index__)(lua_State *L)
{
  if(lua_isnumber(L, 2))
  {
    STORAGE *storage = luaT_checkudata(L, 1, LUA_STORAGE);
    long index = luaL_checklong(L, 2) - 1;
    luaL_argcheck(L, 0 <= index && index < storage->size, 2, "index out of range");
    lua_pushnumber(L, storage->data[index]);
    lua_pushboolean(L, 1);
    return 2;
  }
  else
  {
    lua_pushboolean(L, 0);
    return 1;
  }
}

#ifdef CHAR_STORAGE_STRING
static int W_STORAGE_FUNC(string)(lua_State *L)
{
  THCharStorage *storage = luaT_checkudata(L, 1, torch_CharStorage_id);
  if(lua_isstring(L, -1))
  {
    size_t len = 0;
    const char *str = lua_tolstring(L, -1, &len);
    THCharStorage_resize(storage, len, 0);
    memmove(storage->data, str, len);
    lua_settop(L, 1);
  }
  else
    lua_pushlstring(L, storage->data, storage->size);

  return 1; /* either storage or string */
}
#endif

#define IO_FUNC_TN_(PREFIX,TYPE) PREFIX##TYPE
#define IO_FUNC_TN(PREFIX, TYPE) IO_FUNC_TN_(PREFIX, TYPE)
#define IO_FUNC(PREFIX) IO_FUNC_TN(PREFIX, CAP_TYPE)

static int W_STORAGE_FUNC(factory)(lua_State *L)
{
  STORAGE *storage = STORAGE_FUNC(new)();
  luaT_pushudata(L, storage, LUA_STORAGE);
  return 1;
}

static int W_STORAGE_FUNC(write)(lua_State *L)
{
  STORAGE *storage = luaT_checkudata(L, 1, LUA_STORAGE);
  lua_pushvalue(L, 2);
  torch_File_writeLong(L, &storage->size, 1);
  IO_FUNC(torch_File_write)(L, storage->data, storage->size);
  return 0;
}

static int W_STORAGE_FUNC(read)(lua_State *L)
{
  STORAGE *storage = luaT_checkudata(L, 1, LUA_STORAGE);
  int version = luaL_checkint(L, 3);
  lua_pushvalue(L, 2);
  if(version > 0)
    torch_File_readLong(L, &storage->size, 1);
  else
  {
    int size_;
    torch_File_readInt(L, &size_, 1);
    storage->size = size_;
  }
  STORAGE_FUNC(resize)(storage, storage->size, 0);
  IO_FUNC(torch_File_read)(L, storage->data, storage->size);
  return 0;
}

static const struct luaL_Reg W_STORAGE_FUNC(_) [] = {
  {"size", W_STORAGE_FUNC(__len__)},
  {"__len__", W_STORAGE_FUNC(__len__)},
  {"__newindex__", W_STORAGE_FUNC(__newindex__)},
  {"__index__", W_STORAGE_FUNC(__index__)},
  {"resize", W_STORAGE_FUNC(resize)},
  {"fill", W_STORAGE_FUNC(fill)},
  {"copy", W_STORAGE_FUNC(copy)},
  {"write", W_STORAGE_FUNC(write)},
  {"read", W_STORAGE_FUNC(read)},
#ifdef CHAR_STORAGE_STRING
  {"string", W_STORAGE_FUNC(string)},
#endif
  {NULL, NULL}
};

void W_STORAGE_FUNC(init)(lua_State *L)
{
  LUA_STORAGE = luaT_newmetatable(L, LUA_STORAGE_NAME_T(CAP_TYPE), NULL,
                                  W_STORAGE_FUNC(new), W_STORAGE_FUNC(free), W_STORAGE_FUNC(factory));
  luaL_register(L, NULL, W_STORAGE_FUNC(_));
  lua_pop(L, 1);
}
