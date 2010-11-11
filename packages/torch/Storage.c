#include "Storage.h"
#include "File.h"

static const void *torch_ByteStorage_id = NULL;
static const void *torch_CharStorage_id = NULL;
static const void *torch_ShortStorage_id = NULL;
static const void *torch_IntStorage_id = NULL;
static const void *torch_LongStorage_id = NULL;
static const void *torch_FloatStorage_id = NULL;
static const void *torch_DoubleStorage_id = NULL;

#define TYPE unsigned char
#define CAP_TYPE Byte
#include "StorageGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE char
#define CAP_TYPE Char
#define CHAR_STORAGE_STRING
#include "StorageGen.c"
#undef TYPE
#undef CAP_TYPE
#undef CHAR_STORAGE_STRING

#define TYPE short
#define CAP_TYPE Short
#include "StorageGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE int
#define CAP_TYPE Int
#include "StorageGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE long
#define CAP_TYPE Long
#include "StorageGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE float
#define CAP_TYPE Float
#include "StorageGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE double
#define CAP_TYPE Double
#include "StorageGen.c"
#undef TYPE
#undef CAP_TYPE
