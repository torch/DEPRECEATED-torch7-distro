#ifndef TH_FILE_INC
#define TH_FILE_INC

typedef struct THFile__ THFile;

int THFile_isQuiet(THFile *self);
int THFile_isReadable(THFile *self);
int THFile_isWritable(THFile *self);
int THFile_isBinary(THFile *self);
int THFile_isAutoSpacing(THFile *self);
int THFile_hasError(THFile *self);

long THFile_readByte(THFile *self, unsigned char *data, long n);
long THFile_readChar(THFile *self, char *data, long n);
long THFile_readShort(THFile *self, short *data, long n);
long THFile_readInt(THFile *self, int *data, long n);
long THFile_readLong(THFile *self, long *data, long n);
long THFile_readFloat(THFile *self, float *data, long n);
long THFile_readDouble(THFile *self, double *data, long n);
long THFile_readString(THFile *self, const char *format, char **str_);

long THFile_writeByte(THFile *self, unsigned char *data, long n);
long THFile_writeChar(THFile *self, char *data, long n);
long THFile_writeShort(THFile *self, short *data, long n);
long THFile_writeInt(THFile *self, int *data, long n);
long THFile_writeLong(THFile *self, long *data, long n);
long THFile_writeFloat(THFile *self, float *data, long n);
long THFile_writeDouble(THFile *self, double *data, long n);
long THFile_writeString(THFile *self, const char *str, long size);

void THFile_synchronize(THFile *self);
void THFile_seek(THFile *self, long position);
void THFile_seekEnd(THFile *self);
long THFile_position(THFile *self);
void THFile_close(THFile *self);
void THFile_free(THFile *self);

#endif
