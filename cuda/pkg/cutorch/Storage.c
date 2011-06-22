#include "THCStorage.h"
#include "luaT.h"

static const void *torch_Storage_id;

static int torch_CudaStorage_new(lua_State *L)
{
  THCudaStorage *storage;
  if(lua_type(L, 1) == LUA_TTABLE)
  {
    long size = lua_objlen(L, 1);
    long i;
    storage = THCudaStorage_newWithSize(size);
    for(i = 1; i <= size; i++)
    {
      lua_rawgeti(L, 1, i);
      if(!lua_isnumber(L, -1))
      {
        THCudaStorage_free(storage);
        luaL_error(L, "element at index %d is not a number", i);
      }
      storage->data[i-1] = (real)lua_tonumber(L, -1);
      lua_pop(L, 1);
    }
  }
  else
  {
    long size = luaL_optlong(L, 1, 0);
    storage = THCudaStorage_newWithSize(size);
  }
  luaT_pushudata(L, storage, torch_Storage_id);
  return 1;
}

static int torch_CudaStorage_free(lua_State *L)
{
  THCudaStorage *storage = luaT_checkudata(L, 1, torch_Storage_id);
  THCudaStorage_free(storage);
  return 0;
}

/*
static int torch_CudaStorage_resize(lua_State *L)
{
  THCudaStorage *storage = luaT_checkudata(L, 1, torch_Storage_id);
  long size = luaL_checklong(L, 2);
  THCudaStorage_resize(storage, size);
  lua_settop(L, 1);
  return 1;
}

static int torch_CudaStorage_copy(lua_State *L)
{
  THCudaStorage *storage = luaT_checkudata(L, 1, torch_Storage_id);
  void *src;
  if( (src = luaT_toudata(L, 2, torch_Storage_id)) )
    THCudaStorage_copy(storage, src);
  else if( (src = luaT_toudata(L, 2, torch_ByteStorage_id)) )
    THCudaStorage_copyByte(storage, src);
  else if( (src = luaT_toudata(L, 2, torch_CharStorage_id)) )
    THCudaStorage_copyChar(storage, src);
  else if( (src = luaT_toudata(L, 2, torch_ShortStorage_id)) )
    THCudaStorage_copyShort(storage, src);
  else if( (src = luaT_toudata(L, 2, torch_IntStorage_id)) )
    THCudaStorage_copyInt(storage, src);
  else if( (src = luaT_toudata(L, 2, torch_LongStorage_id)) )
    THCudaStorage_copyLong(storage, src);
  else if( (src = luaT_toudata(L, 2, torch_FloatStorage_id)) )
    THCudaStorage_copyFloat(storage, src);
  else if( (src = luaT_toudata(L, 2, torch_DoubleStorage_id)) )
    THCudaStorage_copyDouble(storage, src);
  else
    luaL_typerror(L, 2, "torch.*Storage");
  lua_settop(L, 1);
  return 1;
}

static int torch_CudaStorage_fill(lua_State *L)
{
  THCudaStorage *storage = luaT_checkudata(L, 1, torch_Storage_id);
  double value = luaL_checknumber(L, 2);
  THCudaStorage_fill(storage, (real)value);
  lua_settop(L, 1);
  return 1;
}

*/
static int torch_CudaStorage___len__(lua_State *L)
{
  THCudaStorage *storage = luaT_checkudata(L, 1, torch_Storage_id);
  lua_pushnumber(L, storage->size);
  return 1;
}

/*
static int torch_CudaStorage___newindex__(lua_State *L)
{
  if(lua_isnumber(L, 2))
  {
    THCudaStorage *storage = luaT_checkudata(L, 1, torch_Storage_id);
    long index = luaL_checklong(L, 2) - 1;
    double number = luaL_checknumber(L, 3);
    luaL_argcheck(L, 0 <= index && index < storage->size, 2, "index out of range");
    storage->data[index] = (real)number;
    lua_pushboolean(L, 1);
  }
  else
    lua_pushboolean(L, 0);

  return 1;
}

static int torch_CudaStorage___index__(lua_State *L)
{
  if(lua_isnumber(L, 2))
  {
    THCudaStorage *storage = luaT_checkudata(L, 1, torch_Storage_id);
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

#if defined(TH_REAL_IS_CHAR) || defined(TH_REAL_IS_BYTE)
static int torch_CudaStorage_string(lua_State *L)
{
  THCudaStorage *storage = luaT_checkudata(L, 1, torch_Storage_id);
  if(lua_isstring(L, -1))
  {
    size_t len = 0;
    const char *str = lua_tolstring(L, -1, &len);
    THCudaStorage_resize(storage, len);
    memmove(storage->data, str, len);
    lua_settop(L, 1);
  }
  else
    lua_pushlstring(L, (char*)storage->data, storage->size);

  return 1;
}
#endif

*/

static int torch_CudaStorage_factory(lua_State *L)
{
  THCudaStorage *storage = THCudaStorage_new();
  luaT_pushudata(L, storage, torch_Storage_id);
  return 1;
}

/*
static int torch_CudaStorage_write(lua_State *L)
{
  THCudaStorage *storage = luaT_checkudata(L, 1, torch_Storage_id);
  lua_pushvalue(L, 2);
  torch_File_writeLong(L, &storage->size, 1);
  torch_File_writeReal(L, storage->data, storage->size);
  return 0;
}

static int torch_CudaStorage_read(lua_State *L)
{
  THCudaStorage *storage = luaT_checkudata(L, 1, torch_Storage_id);
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
  THCudaStorage_resize(storage, storage->size);
  torch_File_readReal(L, storage->data, storage->size);
  return 0;
}
*/

static const struct luaL_Reg torch_CudaStorage__ [] = {
  {"size", torch_CudaStorage___len__},
  {NULL, NULL}
};

void torch_CudaStorage_init(lua_State *L)
{
  torch_Storage_id = luaT_newmetatable(L, "torch.CudaStorage", NULL,
                                       torch_CudaStorage_new, torch_CudaStorage_free, torch_CudaStorage_factory);

  luaL_register(L, NULL, torch_CudaStorage__);
  lua_pop(L, 1);
}
