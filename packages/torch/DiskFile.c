#include "DiskFile.h"

static const void *torch_DiskFile_id = NULL;
static const void *torch_ByteStorage_id = NULL;
static const void *torch_CharStorage_id = NULL;
static const void *torch_ShortStorage_id = NULL;
static const void *torch_IntStorage_id = NULL;
static const void *torch_LongStorage_id = NULL;
static const void *torch_FloatStorage_id = NULL;
static const void *torch_DoubleStorage_id = NULL;

#define READ_WRITE_METHODS(TYPE, TYPEC, ASCII_READ_ELEM, ASCII_WRITE_ELEM) \
static int torch_DiskFile_read##TYPEC(lua_State *L) \
{ \
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id); \
  int nArg = lua_gettop(L); \
  long result = 0L; \
  long n = 1L; \
  TYPE value; \
  TYPE *array = &value; \
  TH##TYPEC##Storage *storage = NULL; \
  luaL_argcheck(L, file->handle, 1, "attempt to use a closed file"); \
  luaL_argcheck(L, file->flags.isReadable, 1, "attempt to read in a write-only file"); \
\
  if(nArg == 2) \
  { \
    if(lua_isnumber(L, 2)) \
    { \
      n = (long)lua_tonumber(L, 2); \
      storage = TH##TYPEC##Storage_newWithSize(n); \
      array = storage->data; \
    } \
    else if( (storage = luaT_toudata(L, 2, torch_##TYPEC##Storage_id)) ) \
    { \
      n = storage->size; \
      array = storage->data; \
    } \
    else \
      luaL_argcheck(L, 0, 2, "number or torch." #TYPEC "Storage expected"); \
  } \
  else if(nArg != 1) \
    luaL_error(L, "bad arguments: should be either nothing, a number, or a torch." #TYPEC "Storage"); \
\
  if(file->flags.isBinary) \
  { \
    result = fread(array, sizeof(TYPE), n, file->handle); \
    if(!file->isNativeEncoding && (sizeof(TYPE) > 1) && (result > 0)) \
      torch_DiskFile_c_reverseMemory(array, array, sizeof(TYPE), result); \
  } \
  else \
  { \
    long i; \
    for(i = 0; i < n; i++) \
    { \
      ASCII_READ_ELEM; /* increment here result and break if wrong */ \
    } \
    if(file->flags.isAutoSpacing && (n > 0)) \
    { \
      int c = fgetc(file->handle); \
      if( (c != '\n') && (c != EOF) ) \
        ungetc(c, file->handle); \
    } \
  } \
\
  if(result != n) \
  { \
    file->flags.hasError = 1; /* shouldn't we put hasError to 0 all the time ? */ \
    if(file->flags.isQuiet) \
    { \
      if(storage && !luaT_toudata(L, 2, torch_##TYPEC##Storage_id)) /* resize if it is mine */ \
        TH##TYPEC##Storage_resize(storage, (result > 0 ? result : 0), 1); \
    } \
    else \
    { \
      if(storage && !luaT_toudata(L, 2, torch_##TYPEC##Storage_id)) /* free if it is mine */ \
        TH##TYPEC##Storage_free(storage); \
      luaL_error(L, "read error: read %d blocks instead of %d", result, n); \
    } \
  } \
  if(storage) \
  { \
    if(luaT_toudata(L, 2, torch_##TYPEC##Storage_id)) /* not mine: i push how much i read */ \
      lua_pushnumber(L, result); \
    else \
      luaT_pushudata(L, storage, torch_##TYPEC##Storage_id); /* mine: i push it */ \
  } \
  else \
    lua_pushnumber(L, value); \
  return 1; \
} \
\
static int torch_DiskFile_write##TYPEC(lua_State *L) \
{ \
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id); \
  long result = 0L; \
  long n = 1L; \
  TYPE value; \
  TYPE *array = &value; \
\
  luaL_argcheck(L, file->handle, 1, "attempt to use a closed file"); \
  luaL_argcheck(L, file->flags.isWritable, 1, "attempt to write in a read-only file"); \
\
  if(lua_isnumber(L, 2)) \
  { \
    value = (TYPE)lua_tonumber(L, 2); \
  } \
  else if(luaT_toudata(L, 2, torch_##TYPEC##Storage_id)) \
  { \
    TH##TYPEC##Storage *storage = luaT_toudata(L, 2, torch_##TYPEC##Storage_id); \
    n = storage->size; \
    array = storage->data; \
  } \
  else \
    luaL_argcheck(L, 0, 2, "number or torch." #TYPEC "Storage expected"); \
\
  if(file->flags.isBinary) \
  { \
    if(file->isNativeEncoding) \
    { \
      result = fwrite(array, sizeof(TYPE), n, file->handle); \
    } \
    else \
    { \
      if(sizeof(TYPE) > 1)						\
      {								        \
	char *buffer = luaT_alloc(L, sizeof(TYPE)*n);			\
	torch_DiskFile_c_reverseMemory(buffer, array, sizeof(TYPE), n); \
	result = fwrite(buffer, sizeof(TYPE), n, file->handle);		\
	luaT_free(L, buffer);						\
      }									\
      else								\
	result = fwrite(array, sizeof(TYPE), n, file->handle);		\
    }									\
  } \
  else \
  { \
    long i; \
    for(i = 0; i < n; i++) \
    { \
      ASCII_WRITE_ELEM; \
      if( file->flags.isAutoSpacing && (i < n-1) ) \
        fprintf(file->handle, " "); \
    } \
    if(file->flags.isAutoSpacing && (n > 0)) \
      fprintf(file->handle, "\n"); \
  } \
\
  if(result != n) \
  { \
    file->flags.hasError = 1; \
    if(!file->flags.isQuiet) \
      luaL_error(L, "write error: wrote %d blocks instead of %d", result, n); \
  } \
  lua_pushnumber(L, result); \
  return 1; \
}

static int torch_DiskFile_c_mode(const char *mode, int *isReadable, int *isWritable)
{
  *isReadable = 0;
  *isWritable = 0;
  if(strlen(mode) == 1)
  {
    if(*mode == 'r')
    {
      *isReadable = 1;
      return 1;
    }
    else if(*mode == 'w')
    {
      *isWritable = 1;
      return 1;
    }
  }
  else if(strlen(mode) == 2)
  {
    if(mode[0] == 'r' && mode[1] == 'w')
    {
      *isReadable = 1;
      *isWritable = 1;
      return 1;
    }
  }
  return 0;
}

static int torch_DiskFile_new(lua_State *L)
{
  const char *name = luaL_checkstring(L, 1);
  const char *mode = luaL_optstring(L, 2, "r");
  int isQuiet = luaT_optboolean(L, 3, 0);
  int isReadable;
  int isWritable;
  FILE *handle;
  DiskFile *file;

  luaL_argcheck(L, torch_DiskFile_c_mode(mode, &isReadable, &isWritable), 2, "file mode should be 'r','w' or 'rw'");

  if( isReadable && isWritable )
  {
    handle = fopen(name, "r+b");
    if(!handle)
    {
      handle = fopen(name, "wb");
      if(handle)
      {
        fclose(handle);
        handle = fopen(name, "r+b");
      }
    }
  }
  else
    handle = fopen(name, (isReadable ? "rb" : "wb"));

  if(!handle)
  {
    if(isQuiet)
      return 0;
    else
      luaL_error(L, "cannot open <%s> in mode %c%c", name, (isReadable ? 'r' : ' '), (isWritable ? 'w' : ' '));
  }

  file = luaT_alloc(L, sizeof(DiskFile));
  file->handle = handle;
  file->flags.isQuiet = isQuiet;
  file->flags.isReadable = isReadable;
  file->flags.isWritable = isWritable;
  file->isNativeEncoding = 1;
  file->flags.isBinary = 0;
  file->flags.isAutoSpacing = 1;
  file->flags.hasError = 0;
  file->name = luaT_alloc(L, strlen(name)+1);
  strcpy(file->name, name);

  luaT_pushudata(L, file, torch_DiskFile_id);
  return 1;
}

static int torch_DiskFile_synchronize(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id);
  if(!file->handle)
    luaL_error(L, "attempt to use a closed file");
  fflush(file->handle);
  lua_settop(L, 1);
  return 1;
}

static int torch_DiskFile_seek(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id);
  long position = luaL_checklong(L, 2)-1;

  luaL_argcheck(L, file->handle, 1, "attempt to use a closed file");
  luaL_argcheck(L, position >= 0, 2, "position must be positive");

  if(fseek(file->handle, position, SEEK_SET) < 0)
  {
    file->flags.hasError = 1;
    if(!file->flags.isQuiet)
      luaL_error(L, "unable to seek at position %d", position);
  }
  lua_settop(L, 1);
  return 1;
}

static int torch_DiskFile_seekEnd(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id);

  luaL_argcheck(L, file->handle, 1, "attempt to use a closed file");

  if(fseek(file->handle, 0L, SEEK_END) < 0)
  {
    file->flags.hasError = 1;
    if(!file->flags.isQuiet)
      luaL_error(L, "unable to seek at end of file");
  }
  lua_settop(L, 1);
  return 1;
}

static int torch_DiskFile_position(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id);
  luaL_argcheck(L, file->handle, 1, "attempt to use a closed file");
  lua_pushnumber(L, ftell(file->handle)+1);
  return 1;
}

static int torch_DiskFile_close(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id);
  luaL_argcheck(L, file->handle, 1, "attempt to use a closed file");
  fclose(file->handle);
  file->handle = NULL;
  return 0;
}

/* Little and Big Endian */

static void torch_DiskFile_c_reverseMemory(void *dst, const void *src, long blockSize, long numBlocks)
{
  if(blockSize != 1)
  {
    long halfBlockSize = blockSize/2;
    char *charSrc = (char*)src;
    char *charDst = (char*)dst;
    long b, i;
    for(b = 0; b < numBlocks; b++)
    {
      for(i = 0; i < halfBlockSize; i++)
      {
        char z = charSrc[i];
        charDst[i] = charSrc[blockSize-1-i];
        charDst[blockSize-1-i] = z;
      }
      charSrc += blockSize;
      charDst += blockSize;
    }
  }
}

static int torch_DiskFile_c_isLittleEndianCPU()
{
  int x = 7;
  char *ptr = (char *)&x;

  if(ptr[0] == 0)
    return 0;
  else
    return 1;
}

static int torch_DiskFile_isLittleEndianCPU(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id);
  luaL_argcheck(L, file->handle, 1, "attempt to use a closed file");
  lua_pushboolean(L, torch_DiskFile_c_isLittleEndianCPU());
  return 1;
}

static int torch_DiskFile_isBigEndianCPU(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id);
  luaL_argcheck(L, file->handle, 1, "attempt to use a closed file");
  lua_pushboolean(L, !torch_DiskFile_c_isLittleEndianCPU());
  return 1;
}

static int torch_DiskFile_nativeEndianEncoding(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id);
  luaL_argcheck(L, file->handle, 1, "attempt to use a closed file");
  file->isNativeEncoding = 1;
  lua_settop(L, 1);
  return 1;
}

static int torch_DiskFile_littleEndianEncoding(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id);
  luaL_argcheck(L, file->handle, 1, "attempt to use a closed file");
  file->isNativeEncoding = torch_DiskFile_c_isLittleEndianCPU();
  lua_settop(L, 1);
  return 1;
}

static int torch_DiskFile_bigEndianEncoding(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id);
  luaL_argcheck(L, file->handle, 1, "attempt to use a closed file");
  file->isNativeEncoding = !torch_DiskFile_c_isLittleEndianCPU();
  lua_settop(L, 1);
  return 1;
}

/* End of Little and Big Endian Stuff */

static int torch_DiskFile_free(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id);
  if(file->handle)
    fclose(file->handle);
  luaT_free(L, file->name);
  luaT_free(L, file);
  return 0;
}

static int torch_DiskFile___tostring__(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id);
  lua_pushfstring(L, "torch.DiskFile on <%s> [status: %s -- mode %c%c]", file->name, (file->handle ? "open" : "closed"),
                  (file->flags.isReadable ? 'r' : ' '), (file->flags.isWritable ? 'w' : ' '));

  return 1;
}

/* READ_WRITE_METHODS(int, Bool, */
/*                    int value = 0; int ret = fscanf(file->handle, "%d", &value); array[i] = (value ? 1 : 0); if(ret <= 0) break; else result++, */
/*                    int value = (array[i] ? 1 : 0); nElemWritten = fprintf(file->handle, "%d", value), */
/*                    true) */

/* Note that we do a trick */
READ_WRITE_METHODS(unsigned char, Byte,
                   result = fread(array, 1, n, file->handle); break,
                   result = fwrite(array, 1, n, file->handle); break)

READ_WRITE_METHODS(char, Char,
                   result = fread(array, 1, n, file->handle); break,
                   result = fwrite(array, 1, n, file->handle); break)

READ_WRITE_METHODS(short, Short,
                   int ret = fscanf(file->handle, "%hd", &array[i]); if(ret <= 0) break; else result++,
                   int ret = fprintf(file->handle, "%hd", array[i]); if(ret <= 0) break; else result++)

READ_WRITE_METHODS(int, Int,
                   int ret = fscanf(file->handle, "%d", &array[i]); if(ret <= 0) break; else result++,
                   int ret = fprintf(file->handle, "%d", array[i]); if(ret <= 0) break; else result++)

READ_WRITE_METHODS(long, Long,
                   int ret = fscanf(file->handle, "%ld", &array[i]); if(ret <= 0) break; else result++,
                   int ret = fprintf(file->handle, "%ld", array[i]); if(ret <= 0) break; else result++)

READ_WRITE_METHODS(float, Float,
                   int ret = fscanf(file->handle, "%g", &array[i]); if(ret <= 0) break; else result++,
                   int ret = fprintf(file->handle, "%g", array[i]); if(ret <= 0) break; else result++)

READ_WRITE_METHODS(double, Double,
                   int ret = fscanf(file->handle, "%lg", &array[i]); if(ret <= 0) break; else result++,
                   int ret = fprintf(file->handle, "%lg", array[i]); if(ret <= 0) break; else result++)

static int torch_DiskFile_readString(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id);
  const char *format = luaL_checkstring(L, 2);
  luaL_argcheck(L, (strlen(format) >= 2 ? (format[0] == '*') && (format[1] == 'a' || format[1] == 'l') : 0), 2, "format must be '*a' or '*l'");
/* note: the string won't survive long, as it is copied into lua */
/* so 1024 is not that big... */
#define TBRS_BSZ 1024L

  if(format[1] == 'a')
  {
    char *p = luaT_alloc(L, TBRS_BSZ);
    long total = TBRS_BSZ;
    long pos = 0L;
    
    for (;;)
    {
      if(total-pos == 0) /* we need more space! */
      {
        total += TBRS_BSZ;
        p = luaT_realloc(L, p, total);
      }
      pos += fread(p+pos, 1, total-pos, file->handle);
      if (pos < total) /* eof? */
      {
        if(pos == 0L)
        {
          luaT_free(L, p);
          file->flags.hasError = 1;
          if(!file->flags.isQuiet)
            luaL_error(L, "read error: read 0 blocks instead of 1");
          return 0;
        }
        lua_pushlstring(L, p, pos);
        luaT_free(L, p);
        return 1;
      }
    }    
  }
  else
  {
    char *p = luaT_alloc(L, TBRS_BSZ);
    long total = TBRS_BSZ;
    long pos = 0L;
    long size;

    for (;;)
    {
      if(total-pos <= 1) /* we can only write '\0' in there! */
      {
        total += TBRS_BSZ;
        p = luaT_realloc(L, p, total);
      }
      if (fgets(p+pos, total-pos, file->handle) == NULL) /* eof? */
      {
        if(pos == 0L)
        {
          luaT_free(L, p);
          file->flags.hasError = 1;
          if(!file->flags.isQuiet)
            luaL_error(L, "read error: read 0 blocks instead of 1");
          return 0;
        }
        lua_pushlstring(L, p, pos);
        luaT_free(L, p);
        return 1;
      }
      size = strlen(p+pos);
      if (size == 0L || (p+pos)[size-1] != '\n')
      {
        pos += size;
      }
      else
      {
        pos += size-1L; /* do not include `eol' */
        lua_pushlstring(L, p, pos);
        luaT_free(L, p);
        return 1;
      }
    }
  }
  return 0;
}


static int torch_DiskFile_writeString(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_DiskFile_id);
  const char *str = NULL;
  size_t size;
  long result;

  luaL_argcheck(L, file->handle, 1, "attempt to use a closed file");
  luaL_argcheck(L, file->flags.isWritable, 1, "attempt to write in a read-only file");
  luaL_checktype(L, 2, LUA_TSTRING);

  str = lua_tolstring(L, 2, &size);
  result = fwrite(str, 1, size, file->handle);
  if(result != (long)size)
  {
    file->flags.hasError = 1;
    if(!file->flags.isQuiet)
      luaL_error(L, "write error: wrote %ld blocks instead of %ld", result, (long)size);
  }
  lua_pushnumber(L, result);
  return 1;
}

static const struct luaL_Reg torch_DiskFile__ [] = {
  {"readByte", torch_DiskFile_readByte},
  {"readChar", torch_DiskFile_readChar},
  {"readShort", torch_DiskFile_readShort},
  {"readInt", torch_DiskFile_readInt},
  {"readLong", torch_DiskFile_readLong},
  {"readFloat", torch_DiskFile_readFloat},
  {"readDouble", torch_DiskFile_readDouble},
  {"readString", torch_DiskFile_readString},
  {"writeByte", torch_DiskFile_writeByte},
  {"writeChar", torch_DiskFile_writeChar},
  {"writeShort", torch_DiskFile_writeShort},
  {"writeInt", torch_DiskFile_writeInt},
  {"writeLong", torch_DiskFile_writeLong},
  {"writeFloat", torch_DiskFile_writeFloat},
  {"writeDouble", torch_DiskFile_writeDouble},
  {"writeString", torch_DiskFile_writeString},
  {"synchronize", torch_DiskFile_synchronize},
  {"seek", torch_DiskFile_seek},
  {"seekEnd", torch_DiskFile_seekEnd},
  {"position", torch_DiskFile_position},
  {"close", torch_DiskFile_close},
  {"isLittleEndianCPU", torch_DiskFile_isLittleEndianCPU},
  {"isBigEndianCPU", torch_DiskFile_isBigEndianCPU},
  {"nativeEndianEncoding", torch_DiskFile_nativeEndianEncoding},
  {"littleEndianEncoding", torch_DiskFile_littleEndianEncoding},
  {"bigEndianEncoding", torch_DiskFile_bigEndianEncoding},
  {"__tostring__", torch_DiskFile___tostring__},
  {NULL, NULL}
};

void torch_DiskFile_init(lua_State *L)
{
  torch_ByteStorage_id = luaT_checktypename2id(L, "torch.ByteStorage");
  torch_CharStorage_id = luaT_checktypename2id(L, "torch.CharStorage");
  torch_ShortStorage_id = luaT_checktypename2id(L, "torch.ShortStorage");
  torch_IntStorage_id = luaT_checktypename2id(L, "torch.IntStorage");
  torch_LongStorage_id = luaT_checktypename2id(L, "torch.LongStorage");
  torch_FloatStorage_id = luaT_checktypename2id(L, "torch.FloatStorage");
  torch_DoubleStorage_id = luaT_checktypename2id(L, "torch.DoubleStorage");

  torch_DiskFile_id = luaT_newmetatable(L, "torch.DiskFile", "torch.File",
                                        torch_DiskFile_new, torch_DiskFile_free, NULL);

  luaL_register(L, NULL, torch_DiskFile__);
  lua_pop(L, 1);
}
