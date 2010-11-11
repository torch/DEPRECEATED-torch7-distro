#include "Tensor.h"
#include "File.h"

static const void *torch_ByteStorage_id = NULL;
static const void *torch_CharStorage_id = NULL;
static const void *torch_ShortStorage_id = NULL;
static const void *torch_IntStorage_id = NULL;
static const void *torch_LongStorage_id = NULL;
static const void *torch_FloatStorage_id = NULL;
static const void *torch_DoubleStorage_id = NULL;

static const void *torch_ByteTensor_id = NULL;
static const void *torch_CharTensor_id = NULL;
static const void *torch_ShortTensor_id = NULL;
static const void *torch_IntTensor_id = NULL;
static const void *torch_LongTensor_id = NULL;
static const void *torch_FloatTensor_id = NULL;
static const void *torch_Tensor_id = NULL;

#define TYPE unsigned char
#define CAP_TYPE Byte
#include "TensorGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE char
#define CAP_TYPE Char
#include "TensorGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE short
#define CAP_TYPE Short
#include "TensorGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE int
#define CAP_TYPE Int
#include "TensorGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE long
#define CAP_TYPE Long
#include "TensorGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE float
#define CAP_TYPE Float
#include "TensorGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE double
#define CAP_TYPE Double
#define DEFAULT_TENSOR
#include "TensorGen.c"
#undef TYPE
#undef CAP_TYPE
#undef DEFAULT_TENSOR
