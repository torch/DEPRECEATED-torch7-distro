#include "THFile.h"
#include "THFilePrivate.h"

#define IMPLEMENT_THFILE(TYPEC, TYPE)                         \
  long THFile_read##TYPEC(THFile *self, TYPE *data, long n)   \
  {                                                           \
    return (*self->vtable->read##TYPEC)(self, data, n);       \
  }                                                           \
                                                              \
  long THFile_write##TYPEC(THFile *self, TYPE *data, long n)  \
  {                                                           \
    return (*self->vtable->write##TYPEC)(self, data, n);      \
  }
  
IMPLEMENT_THFILE(Byte, unsigned char)
IMPLEMENT_THFILE(Char, char)
IMPLEMENT_THFILE(Short, short)
IMPLEMENT_THFILE(Int, int)
IMPLEMENT_THFILE(Long, long)
IMPLEMENT_THFILE(Float, float)
IMPLEMENT_THFILE(Double, double)

long THFile_readString(THFile *self, const char *format, char **str_)
{
  return self->vtable->readString(self, format, str_);
}

long THFile_writeString(THFile *self, const char *str, long size)
{
  return self->vtable->writeString(self, str, size);
}

void THFile_synchronize(THFile *self)
{
  self->vtable->synchronize(self);
}

void THFile_seek(THFile *self, long position)
{
  self->vtable->seek(self, position);
}

void THFile_seekEnd(THFile *self)
{
  self->vtable->seekEnd(self);
}

long THFile_position(THFile *self)
{
  return self->vtable->position(self);
}

void THFile_close(THFile *self)
{
  self->vtable->close(self);
}

void THFile_free(THFile *self)
{
  self->vtable->free(self);
}

#define IMPLEMENT_THFILE_FLAGS(FLAG) \
  int THFile_##FLAG(THFile *self)    \
  {                                  \
    return self->FLAG;               \
  }

IMPLEMENT_THFILE_FLAGS(isReadable)
IMPLEMENT_THFILE_FLAGS(isWritable)
IMPLEMENT_THFILE_FLAGS(isBinary)
IMPLEMENT_THFILE_FLAGS(isAutoSpacing)
IMPLEMENT_THFILE_FLAGS(hasError)
