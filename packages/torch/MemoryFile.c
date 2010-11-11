#include "MemoryFile.h"

static const void* torch_MemoryFile_id = NULL;
static const void *torch_CharStorage_id = NULL;
static const void *torch_ShortStorage_id = NULL;
static const void *torch_IntStorage_id = NULL;
static const void *torch_LongStorage_id = NULL;
static const void *torch_FloatStorage_id = NULL;
static const void *torch_DoubleStorage_id = NULL;

/********************** UTILITIES ***********************/

static char *torch_MemoryFile_c_strnextspace(char *str_, char *c_)
{
  char c;

  while( (c = *str_) )
  {
    if( (c != ' ') && (c != '\n') && (c != ':') && (c != ';') )
      break;
    str_++;
  }

  while( (c = *str_) )
  {
    if( (c == ' ') || (c == '\n') || (c == ':') || (c == ';') )
    {
      *c_ = c;
      *str_ = '\0';
      return(str_);
    }
    str_++;
  }
  return NULL;
}

static void torch_MemoryFile_c_grow(MemoryFile *file, long size)
{
  long missingSpace;

  if(size <= file->size)
    return;
  else
  {
    if(size < file->storage->size) /* note the "<" and not "<=" */
    {
      file->size = size;
      file->storage->data[file->size] = '\0';
      return;
    }
  }

  missingSpace = size-file->storage->size+1; /* +1 for the '\0' */
  THCharStorage_resize(file->storage, (file->storage->size/2 > missingSpace ?
                                       file->storage->size + (file->storage->size/2)
                                       : file->storage->size + missingSpace),
                       1);
}

static int torch_MemoryFile_c_mode(const char *mode, int *isReadable, int *isWritable)
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

/********************************************************/

#define READ_WRITE_METHODS(TYPE, TYPEC, ASCII_READ_ELEM, ASCII_WRITE_ELEM, INSIDE_SPACING) \
static int torch_MemoryFile_read##TYPEC(lua_State *L) \
{ \
  MemoryFile *file = luaT_checkudata(L, 1, torch_MemoryFile_id); \
  int nArg = lua_gettop(L); \
  long result = 0L; \
  long n = 1L; \
  TYPE value; \
  TYPE *array = &value; \
  TH##TYPEC##Storage *storage = NULL; \
  luaL_argcheck(L, file->storage, 1, "attempt to use a closed file"); \
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
    long nByte = sizeof(TYPE)*n; \
    long nByteRemaining = (file->position + nByte <= file->size ? nByte : file->size-file->position); \
    result = nByteRemaining/sizeof(TYPE); \
    memmove(array, file->storage->data+file->position, result*sizeof(TYPE)); \
    file->position += result*sizeof(TYPE); \
  } \
  else \
  { \
    long i; \
    for(i = 0; i < n; i++) \
    { \
      long nByteRead = 0; \
      char spaceChar = 0; \
      char *spacePtr = torch_MemoryFile_c_strnextspace(file->storage->data+file->position, &spaceChar); \
      ASCII_READ_ELEM; \
      if(ret == EOF) \
      { \
        while(file->storage->data[file->position]) \
          file->position++; \
      } \
      else \
         file->position += nByteRead; \
      if(spacePtr) \
        *spacePtr = spaceChar; \
    } \
    if(file->flags.isAutoSpacing && (n > 0)) \
    { \
      if( (file->position < file->size) && (file->storage->data[file->position] == '\n') ) \
        file->position++; \
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
static int torch_MemoryFile_write##TYPEC(lua_State *L) \
{ \
  MemoryFile *file = luaT_checkudata(L, 1, torch_MemoryFile_id); \
  long n = 1L; \
  TYPE value; \
  TYPE *array = &value; \
\
  luaL_argcheck(L, file->storage, 1, "attempt to use a closed file"); \
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
    long nByte = sizeof(TYPE)*n; \
    torch_MemoryFile_c_grow(file, file->position+nByte); \
    memmove(file->storage->data+file->position, array, nByte); \
    file->position += nByte; \
    if(file->position > file->size) \
    { \
      file->size = file->position; \
      file->storage->data[file->size] = '\0'; \
    } \
  } \
  else \
  { \
    long i; \
    for(i = 0; i < n; i++) \
    { \
      long nByteWritten; \
      while (1) \
      { \
        ASCII_WRITE_ELEM; \
        if( (nByteWritten > -1) && (nByteWritten < file->storage->size-file->position) ) \
        { \
          file->position += nByteWritten; \
          break; \
        } \
        torch_MemoryFile_c_grow(file, file->storage->size + (file->storage->size/2) + 2); \
      } \
      if(file->flags.isAutoSpacing) \
      { \
        if(i < n-1) \
        { \
          torch_MemoryFile_c_grow(file, file->position+1); \
          sprintf(file->storage->data+file->position, " "); \
          file->position++; \
        } \
        if(i == n-1) \
        { \
          torch_MemoryFile_c_grow(file, file->position+1); \
          sprintf(file->storage->data+file->position, "\n"); \
          file->position++; \
        } \
      } \
    } \
    if(file->position > file->size) \
    { \
      file->size = file->position; \
      file->storage->data[file->size] = '\0'; \
    } \
  } \
\
  lua_pushnumber(L, n); \
  return 1; \
}

static int torch_MemoryFile_new(lua_State *L)
{
  const char *mode;
  int isReadable;
  int isWritable;
  THCharStorage *storage = luaT_toudata(L, 1, torch_CharStorage_id);
  MemoryFile *file;

  if(storage)
  {
    luaL_argcheck(L, storage->data[storage->size-1] == '\0', 1, "provided CharStorage must be terminated by 0");
    mode = luaL_optstring(L, 2, "rw");
    luaL_argcheck(L, torch_MemoryFile_c_mode(mode, &isReadable, &isWritable), 2, "file mode should be 'r','w' or 'rw'");
    storage->refcount++;
  }
  else
  {
    mode = luaL_optstring(L, 1, "rw");    
    luaL_argcheck(L, torch_MemoryFile_c_mode(mode, &isReadable, &isWritable), 2, "file mode should be 'r','w' or 'rw'");
    storage = THCharStorage_newWithSize(1);
    storage->data[0] = '\0';
  }

  file = luaT_alloc(L, sizeof(MemoryFile));
  file->storage = storage;
  file->size = (storage ? storage->size-1 : 0);
  file->position = 0;
  file->flags.isQuiet = 0;
  file->flags.isReadable = isReadable;
  file->flags.isWritable = isWritable;
  file->flags.isBinary = 0;
  file->flags.isAutoSpacing = 1;
  file->flags.hasError = 0;

  luaT_pushudata(L, file, torch_MemoryFile_id);
  return 1;
}

static int torch_MemoryFile_storage(lua_State *L)
{
  MemoryFile *file = luaT_checkudata(L, 1, torch_MemoryFile_id);
  if(!file->storage)
    luaL_error(L, "attempt to use a closed file");
  THCharStorage_resize(file->storage, file->size+1, 1);
  file->storage->refcount++;
  luaT_pushudata(L, file->storage, torch_CharStorage_id);
  return 1;
}

static int torch_MemoryFile_synchronize(lua_State *L)
{
  MemoryFile *file = luaT_checkudata(L, 1, torch_MemoryFile_id);
  if(!file->storage)
    luaL_error(L, "attempt to use a closed file");
  lua_settop(L, 1);
  return 1;
}

static int torch_MemoryFile_seek(lua_State *L)
{
  MemoryFile *file = luaT_checkudata(L, 1, torch_MemoryFile_id);
  long position = luaL_checklong(L, 2)-1;

  luaL_argcheck(L, file->storage, 1, "attempt to use a closed file");
  luaL_argcheck(L, position >= 0, 2, "position must be positive");

  if(position <= file->size)
    file->position = position;
  else
  {
    file->flags.hasError = 1;
    if(!file->flags.isQuiet)
      luaL_error(L, "unable to seek at position %d", position);
  }
  lua_settop(L, 1);
  return 1;
}

static int torch_MemoryFile_seekEnd(lua_State *L)
{
  MemoryFile *file = luaT_checkudata(L, 1, torch_MemoryFile_id);
  luaL_argcheck(L, file->storage, 1, "attempt to use a closed file");
  file->position = file->size;
  lua_settop(L, 1);
  return 1;
}

static int torch_MemoryFile_position(lua_State *L)
{
  MemoryFile *file = luaT_checkudata(L, 1, torch_MemoryFile_id);
  luaL_argcheck(L, file->storage, 1, "attempt to use a closed file");
  lua_pushnumber(L, file->position+1);
  return 1;
}

static int torch_MemoryFile_close(lua_State *L)
{
  MemoryFile *file = luaT_checkudata(L, 1, torch_MemoryFile_id);
  luaL_argcheck(L, file->storage, 1, "attempt to use a closed file");
  THCharStorage_free(file->storage);
  file->storage = NULL;
  return 0;
}

static int torch_MemoryFile_free(lua_State *L)
{
  MemoryFile *file = luaT_checkudata(L, 1, torch_MemoryFile_id);
  if(file->storage)
    THCharStorage_free(file->storage);
  luaT_free(L, file);
  return 0;
}

/* READ_WRITE_METHODS(bool, Bool, */
/*                    int value = 0; int ret = sscanf(file->storage->data+file->position, "%d%n", &value, &nByteRead); array[i] = (value ? 1 : 0), */
/*                    int value = (array[i] ? 1 : 0); nByteWritten = snprintf(file->storage->data+file->position, file->storage->size-file->position, "%d", value), */
/*                    1) */

/* DEBUG: we should check if %n is count or not as a element (so ret might need to be ret-- on some systems) */
/* Note that we do a trick for char */
READ_WRITE_METHODS(char, Char,
                   long ret = (file->position + n <= file->size ? n : file->size-file->position);  \
                   if(spacePtr) *spacePtr = spaceChar; \
                   nByteRead = ret; \
                   result = ret; \
                   i = n-1; \
                   memmove(array, file->storage->data+file->position, nByteRead),
                   nByteWritten = (n < file->storage->size-file->position ? n : -1); \
                   i = n-1; \
                   if(nByteWritten > -1)
                     memmove(file->storage->data+file->position, array, nByteWritten),
                   0)

READ_WRITE_METHODS(short, Short,
                   int nByteRead_; int ret = sscanf(file->storage->data+file->position, "%hd%n", &array[i], &nByteRead_); nByteRead = nByteRead_; if(ret <= 0) break; else result++,
                   nByteWritten = snprintf(file->storage->data+file->position, file->storage->size-file->position, "%hd", array[i]),
                   1)

READ_WRITE_METHODS(int, Int,
                   int nByteRead_; int ret = sscanf(file->storage->data+file->position, "%d%n", &array[i], &nByteRead_); nByteRead = nByteRead_; if(ret <= 0) break; else result++,
                   nByteWritten = snprintf(file->storage->data+file->position, file->storage->size-file->position, "%d", array[i]),
                   1)

READ_WRITE_METHODS(long, Long,
                   int nByteRead_; int ret = sscanf(file->storage->data+file->position, "%ld%n", &array[i], &nByteRead_); nByteRead = nByteRead_; if(ret <= 0) break; else result++,
                   nByteWritten = snprintf(file->storage->data+file->position, file->storage->size-file->position, "%ld", array[i]),
                   1)

READ_WRITE_METHODS(float, Float,
                   int nByteRead_; int ret = sscanf(file->storage->data+file->position, "%g%n", &array[i], &nByteRead_); nByteRead = nByteRead_; if(ret <= 0) break; else result++,
                   nByteWritten = snprintf(file->storage->data+file->position, file->storage->size-file->position, "%g", array[i]),
                   1)

READ_WRITE_METHODS(double, Double,
                   int nByteRead_; int ret = sscanf(file->storage->data+file->position, "%lg%n", &array[i], &nByteRead_); nByteRead = nByteRead_; if(ret <= 0) break; else result++,
                   nByteWritten = snprintf(file->storage->data+file->position, file->storage->size-file->position, "%lg", array[i]),
                   1)

static int torch_MemoryFile_readString(lua_State *L)
{
  MemoryFile *file = luaT_checkudata(L, 1, torch_MemoryFile_id);
  const char *format = luaL_checkstring(L, 2);
  luaL_argcheck(L, (strlen(format) >= 2 ? (format[0] == '*') && (format[1] == 'a' || format[1] == 'l') : 0), 2, "format must be '*a' or '*l'");

  if(file->position == file->size) /* eof ? */
  {
    file->flags.hasError = 1;
    if(!file->flags.isQuiet)
      luaL_error(L, "read error: read 0 blocks instead of 1");
    return 0;
  }
  
  if(format[1] == 'a')
  {
    lua_pushlstring(L, file->storage->data+file->position, file->size-file->position);
    file->position = file->size;
    return 1;
  }
  else
  {
    char *p = file->storage->data+file->position;
    long posEol = -1;
    long i;
    for(i = 0L; i < file->size-file->position; i++)
    {
      if(p[i] == '\n')
      {
        posEol = i;
        break;
      }
    }

    if(posEol >= 0)
    {
      lua_pushlstring(L, file->storage->data+file->position, posEol); /* do not copy the end of line */
      file->position += posEol+1;
      return 1;
    }
    else /* well, we read all! */
    {
      lua_pushlstring(L, file->storage->data+file->position, file->size-file->position);
      file->position = file->size;
      return 1;
    }
  }
  return 0;
}

static int torch_MemoryFile_writeString(lua_State *L)
{
  MemoryFile *file = luaT_checkudata(L, 1, torch_MemoryFile_id);
  const char *str = NULL;
  size_t size;

  luaL_argcheck(L, file->storage, 1, "attempt to use a closed file");
  luaL_argcheck(L, file->flags.isWritable, 1, "attempt to write in a read-only file");
  luaL_checktype(L, 2, LUA_TSTRING);

  str = lua_tolstring(L, 2, &size);
  torch_MemoryFile_c_grow(file, file->position+size);
  memmove(file->storage->data+file->position, str, size);
  file->position += size;
  if(file->position > file->size)
  {
    file->size = file->position;
    file->storage->data[file->size] = '\0';
  }
  lua_pushnumber(L, size);
  return 1;
}

static int torch_MemoryFile___tostring__(lua_State *L)
{
  MemoryFile *file = luaT_checkudata(L, 1, torch_MemoryFile_id);
  lua_pushfstring(L, "torch.MemoryFile [status: %s -- mode: %c%c]", (file->storage ? "open" : "closed"),
                  (file->flags.isReadable ? 'r' : ' '), (file->flags.isWritable ? 'w' : ' '));
  return 1;
}

static const struct luaL_Reg torch_MemoryFile__ [] = {
  {"readChar", torch_MemoryFile_readChar},
  {"readShort", torch_MemoryFile_readShort},
  {"readInt", torch_MemoryFile_readInt},
  {"readLong", torch_MemoryFile_readLong},
  {"readFloat", torch_MemoryFile_readFloat},
  {"readDouble", torch_MemoryFile_readDouble},
  {"readString", torch_MemoryFile_readString},
  {"writeChar", torch_MemoryFile_writeChar},
  {"writeShort", torch_MemoryFile_writeShort},
  {"writeInt", torch_MemoryFile_writeInt},
  {"writeLong", torch_MemoryFile_writeLong},
  {"writeFloat", torch_MemoryFile_writeFloat},
  {"writeDouble", torch_MemoryFile_writeDouble},
  {"writeString", torch_MemoryFile_writeString},
  {"synchronize", torch_MemoryFile_synchronize},
  {"seek", torch_MemoryFile_seek},
  {"seekEnd", torch_MemoryFile_seekEnd},
  {"position", torch_MemoryFile_position},
  {"close", torch_MemoryFile_close},
  {"storage", torch_MemoryFile_storage},
  {"__tostring__", torch_MemoryFile___tostring__},
  {NULL, NULL}
};

void torch_MemoryFile_init(lua_State *L)
{
  torch_CharStorage_id = luaT_checktypename2id(L, "torch.CharStorage");
  torch_ShortStorage_id = luaT_checktypename2id(L, "torch.ShortStorage");
  torch_IntStorage_id = luaT_checktypename2id(L, "torch.IntStorage");
  torch_LongStorage_id = luaT_checktypename2id(L, "torch.LongStorage");
  torch_FloatStorage_id = luaT_checktypename2id(L, "torch.FloatStorage");
  torch_DoubleStorage_id = luaT_checktypename2id(L, "torch.DoubleStorage");

  torch_MemoryFile_id = luaT_newmetatable(L, "torch.MemoryFile", "torch.File",
                                          torch_MemoryFile_new, torch_MemoryFile_free, NULL);
  luaL_register(L, NULL, torch_MemoryFile__);
  lua_pop(L, 1);
}
