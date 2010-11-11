#ifndef TORCH_FILE_INC
#define TORCH_FILE_INC

#include "general.h"

typedef struct File__
{
    int isQuiet;
    int isReadable;
    int isWritable;
    int isBinary;
    int isAutoSpacing;
    int hasError;
} File;

long torch_File_readByte(lua_State *L, unsigned char *data, long n);
long torch_File_readChar(lua_State *L, char *data, long n);
long torch_File_readShort(lua_State *L, short *data, long n);
long torch_File_readInt(lua_State *L, int *data, long n);
long torch_File_readLong(lua_State *L, long *data, long n);
long torch_File_readFloat(lua_State *L, float *data, long n);
long torch_File_readDouble(lua_State *L, double *data, long n);
long torch_File_readObject(lua_State *L);

long torch_File_writeByte(lua_State *L, unsigned char *data, long n);
long torch_File_writeChar(lua_State *L, char *data, long n);
long torch_File_writeShort(lua_State *L, short *data, long n);
long torch_File_writeInt(lua_State *L, int *data, long n);
long torch_File_writeLong(lua_State *L, long *data, long n);
long torch_File_writeFloat(lua_State *L, float *data, long n);
long torch_File_writeDouble(lua_State *L, double *data, long n);
long torch_File_writeObject(lua_State *L);

void torch_File_init(lua_State *L);

#endif
