#include "Tensor.h"

#define torch_TensorMath_(NAME) TH_CONCAT_4(torch_,Real,TensorMath_,NAME)
#define torch_Tensor_id TH_CONCAT_3(torch_,Real,Tensor_id)
#define STRING_torchTensor TH_CONCAT_STRING_3(torch.,Real,Tensor)

#include "generic/TensorMath.c"
#include "THGenerateAllTypes.h"
