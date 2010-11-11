#ifndef TORCH_MEMORY_FILE_INC
#define TORCH_MEMORY_FILE_INC

#include "File.h"

typedef struct MemoryFile__
{
    File flags;
    THCharStorage *storage;
    long size;
    long position;
} MemoryFile;

void torch_MemoryFile_init(lua_State *L);

#endif
