#ifndef TORCH_DISK_FILE_INC
#define TORCH_DISK_FILE_INC

#include "File.h"

typedef struct DiskFile__
{
    File flags;
    FILE *handle;
    char *name;
    int isNativeEncoding;
} DiskFile;

void torch_DiskFile_init(lua_State *L);

#endif
