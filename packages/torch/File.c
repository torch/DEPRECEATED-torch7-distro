#include "File.h"

static const void *torch_File_id = NULL;
static const void *torch_ByteStorage_id = NULL;
static const void *torch_CharStorage_id = NULL;
static const void *torch_ShortStorage_id = NULL;
static const void *torch_IntStorage_id = NULL;
static const void *torch_LongStorage_id = NULL;
static const void *torch_FloatStorage_id = NULL;
static const void *torch_DoubleStorage_id = NULL;

static int torch_File_binary(lua_State *L)
{
  File *file = luaT_checkudata(L, 1, torch_File_id);
  file->isBinary = 1;
  lua_settop(L, 1);
  return 1;
}

static int torch_File_ascii(lua_State *L)
{
  File *file = luaT_checkudata(L, 1, torch_File_id);
  file->isBinary = 0;
  lua_settop(L, 1);
  return 1;
}

static int torch_File_autoSpacing(lua_State *L)
{
  File *file = luaT_checkudata(L, 1, torch_File_id);
  file->isAutoSpacing = 1;
  lua_settop(L, 1);
  return 1;
}

static int torch_File_noAutoSpacing(lua_State *L)
{
  File *file = luaT_checkudata(L, 1, torch_File_id);
  file->isAutoSpacing = 0;
  lua_settop(L, 1);
  return 1;
}

static int torch_File_quiet(lua_State *L)
{
  File *file = luaT_checkudata(L, 1, torch_File_id);
  file->isQuiet = 1;
  lua_settop(L, 1);
  return 1;
}

static int torch_File_isQuiet(lua_State *L)
{
  File *file = luaT_checkudata(L, 1, torch_File_id);
  lua_pushboolean(L, file->isQuiet);
  return 1;
}

static int torch_File_pedantic(lua_State *L)
{
  File *file = luaT_checkudata(L, 1, torch_File_id);
  file->isQuiet = 0;
  lua_settop(L, 1);
  return 1;
}

static int torch_File_clearError(lua_State *L)
{
  File *file = luaT_checkudata(L, 1, torch_File_id);
  file->hasError = 0;
  lua_settop(L, 1);
  return 1;
}

static int torch_File_hasError(lua_State *L)
{
  File *file = luaT_checkudata(L, 1, torch_File_id);
  lua_pushboolean(L, file->hasError);
  return 1;
}

static int torch_File_isReadable(lua_State *L)
{
  File *file = luaT_checkudata(L, 1, torch_File_id);
  lua_pushboolean(L, file->isReadable);
  return 1;
}

static int torch_File_isWritable(lua_State *L)
{
  File *file = luaT_checkudata(L, 1, torch_File_id);
  lua_pushboolean(L, file->isWritable);
  return 1;
}

static const struct luaL_Reg torch_File__ [] = {
  {"binary", torch_File_binary},
  {"ascii", torch_File_ascii},
  {"autoSpacing", torch_File_autoSpacing},
  {"noAutoSpacing", torch_File_noAutoSpacing},
  {"quiet", torch_File_quiet},
  {"isQuiet", torch_File_isQuiet},
  {"pedantic", torch_File_pedantic},
  {"hasError", torch_File_hasError},
  {"clearError", torch_File_clearError},
  {"isReadable", torch_File_isReadable},
  {"isWritable", torch_File_isWritable},
  {NULL, NULL}
};

void torch_File_init(lua_State *L)
{
  torch_ByteStorage_id = luaT_checktypename2id(L, "torch.ByteStorage");
  torch_CharStorage_id = luaT_checktypename2id(L, "torch.CharStorage");
  torch_ShortStorage_id = luaT_checktypename2id(L, "torch.ShortStorage");
  torch_IntStorage_id = luaT_checktypename2id(L, "torch.IntStorage");
  torch_LongStorage_id = luaT_checktypename2id(L, "torch.LongStorage");
  torch_FloatStorage_id = luaT_checktypename2id(L, "torch.FloatStorage");
  torch_DoubleStorage_id = luaT_checktypename2id(L, "torch.DoubleStorage");

  torch_File_id = luaT_newmetatable(L, "torch.File", NULL, NULL, NULL, NULL);
  luaL_register(L, NULL, torch_File__);
  lua_pop(L, 1);
}

#define IMPLEMENT_TORCH_FILE_READ(TYPE, CTYPE) \
long torch_File_read##CTYPE(lua_State *L, TYPE *data, long n) \
{ \
  static TH##CTYPE##Storage torch_File_##CTYPE##Storage; \
  long ret; \
\
  int index = lua_gettop(L); \
  if(!luaT_isudata(L, index, torch_File_id)) \
    luaL_error(L, "Internal error in write" #CTYPE ": not a File at stack index %d", index); \
  lua_getfield(L, index, "read" #CTYPE); \
  lua_pushvalue(L, index); \
  torch_File_##CTYPE##Storage.data = data; \
  torch_File_##CTYPE##Storage.size = n; \
  torch_File_##CTYPE##Storage.refcount = -1; /* makes sure nobody can free it */ \
  luaT_pushudata(L, &torch_File_##CTYPE##Storage, torch_##CTYPE##Storage_id); \
  lua_call(L, 2, 1); \
  ret = (long)lua_tonumber(L, -1); \
  lua_pop(L, 1); \
  return ret; \
}

#define IMPLEMENT_TORCH_FILE_WRITE(TYPE, CTYPE) \
long torch_File_write##CTYPE(lua_State *L, TYPE *data, long n) \
{ \
  static TH##CTYPE##Storage torch_File_##CTYPE##Storage; \
  long ret; \
\
  int index = lua_gettop(L); \
  if(!luaT_isudata(L, index, torch_File_id)) \
    luaL_error(L, "Internal error in write" #CTYPE ": not a File at stack index %d", index); \
  lua_getfield(L, index, "write" #CTYPE); \
  lua_pushvalue(L, index); \
  torch_File_##CTYPE##Storage.data = data; \
  torch_File_##CTYPE##Storage.size = n; \
  torch_File_##CTYPE##Storage.refcount = -1; /* makes sure nobody can free it */ \
  luaT_pushudata(L, &torch_File_##CTYPE##Storage, torch_##CTYPE##Storage_id); \
  lua_call(L, 2, 1); \
  ret = (long)lua_tonumber(L, -1); \
  lua_pop(L, 1); \
  return ret; \
}

IMPLEMENT_TORCH_FILE_READ(unsigned char, Byte)
IMPLEMENT_TORCH_FILE_READ(char, Char)
IMPLEMENT_TORCH_FILE_READ(short, Short)
IMPLEMENT_TORCH_FILE_READ(int, Int)
IMPLEMENT_TORCH_FILE_READ(long, Long)
IMPLEMENT_TORCH_FILE_READ(float, Float)
IMPLEMENT_TORCH_FILE_READ(double, Double)

IMPLEMENT_TORCH_FILE_WRITE(unsigned char, Byte)
IMPLEMENT_TORCH_FILE_WRITE(char, Char)
IMPLEMENT_TORCH_FILE_WRITE(short, Short)
IMPLEMENT_TORCH_FILE_WRITE(int, Int)
IMPLEMENT_TORCH_FILE_WRITE(long, Long)
IMPLEMENT_TORCH_FILE_WRITE(float, Float)
IMPLEMENT_TORCH_FILE_WRITE(double, Double)

long torch_File_readObject(lua_State *L)
{
  int index = lua_gettop(L);

  if(!luaT_isudata(L, index, torch_File_id))
    luaL_error(L, "Internal error in readObject: not a File at stack index %d", index);
  
  lua_getfield(L, index, "readObject");
  lua_pushvalue(L, index);
  lua_call(L, 1, 1);
  return 1;
}

long torch_File_writeObject(lua_State *L)
{
  int index = lua_gettop(L);

  if(index < 2)
    luaL_error(L, "File and object expected");

  if(!luaT_isudata(L, index-1, torch_File_id))
    luaL_error(L, "Internal error in writeObject: not a File at stack index %d", index-1);

  lua_getfield(L, index-1, "writeObject");
  lua_pushvalue(L, index-1);
  lua_pushvalue(L, index);
  lua_call(L, 2, 0);
  lua_pop(L, 1); /* remove the object from the stack */

  return 1;
}
