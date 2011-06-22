#ifndef THC_STORAGE_INC
#define THC_STORAGE_INC

#include "THGeneral.h"

#define THStorage        TH_CONCAT_3(TH,Cuda,Storage)
#define THStorage_(NAME) TH_CONCAT_4(TH,Cuda,Storage_,NAME)
#define real float

#define TH_GENERIC_FILE "generic/THStorage.h"
#include "generic/THStorage.h"
#undef TH_GENERIC_FILE

#endif
