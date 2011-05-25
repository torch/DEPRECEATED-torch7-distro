#include "Storage.h"
#include "File.h"

static const void *torch_ByteStorage_id = NULL;
static const void *torch_CharStorage_id = NULL;
static const void *torch_ShortStorage_id = NULL;
static const void *torch_IntStorage_id = NULL;
static const void *torch_LongStorage_id = NULL;
static const void *torch_FloatStorage_id = NULL;
static const void *torch_DoubleStorage_id = NULL;

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,Real,Storage_,NAME)
#define torch_Storage_id TH_CONCAT_3(torch_,Real,Storage_id)
#define torch_File_readReal TH_CONCAT_2(torch_File_read, Real)
#define torch_File_writeReal TH_CONCAT_2(torch_File_write, Real)
#define STRING_torchStorage TH_CONCAT_STRING_3(torch.,Real,Storage)

#include "generic/Storage.c"
#include "THGenerateAllTypes.h"
