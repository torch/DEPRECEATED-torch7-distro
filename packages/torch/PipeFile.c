#include "DiskFile.h"

static const void* torch_PipeFile_id = NULL;

static int torch_PipeFile_c_mode(const char *mode, int *isReadable, int *isWritable)
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
  return 0;
}

static int torch_PipeFile_new(lua_State *L)
{
  const char *name = luaL_checkstring(L, 1);
  const char *mode = luaL_optstring(L, 2, "r");
  int isQuiet = luaT_optboolean(L, 3, 0);
  int isReadable;
  int isWritable;
  FILE *handle;
  DiskFile *file;

  luaL_argcheck(L, torch_PipeFile_c_mode(mode, &isReadable, &isWritable), 2, "file mode should be 'r','w'");

#ifdef _WIN32
  handle = popen(name, (isReadable ? "rb" : "wb"));
#else
  handle = popen(name, (isReadable ? "r" : "w"));
#endif

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

  luaT_pushudata(L, file, torch_PipeFile_id);
  return 1;
}

static int torch_PipeFile_free(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_PipeFile_id);
  if(file->handle)
    pclose(file->handle);
  luaT_free(L, file->name);
  luaT_free(L, file);
  return 0;
}

static int torch_PipeFile___tostring__(lua_State *L)
{
  DiskFile *file = luaT_checkudata(L, 1, torch_PipeFile_id);
  lua_pushfstring(L, "torch.PipeFile on <%s> [status: %s -- mode: %c%c]", file->name, (file->handle ? "open" : "closed"),
                  (file->flags.isReadable ? 'r' : ' '), (file->flags.isWritable ? 'w' : ' '));
  return 1;
}


static const struct luaL_Reg torch_PipeFile__ [] = {
  {"__tostring__", torch_PipeFile___tostring__},
  {NULL, NULL}
};

void torch_PipeFile_init(lua_State *L)
{
  torch_PipeFile_id = luaT_newmetatable(L, "torch.PipeFile", "torch.DiskFile",
                                        torch_PipeFile_new, torch_PipeFile_free, NULL);

  luaL_register(L, NULL, torch_PipeFile__);
  lua_pop(L, 1);
}
