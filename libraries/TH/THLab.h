#include "THTensor.h"

#include "THTensor.h"
#include "THVector.h"
#include "THBlas.h"
#include "THLapack.h"

#define THLab_(NAME)   TH_CONCAT_4(TH,Real,Lab_,NAME)

#include "generic/THLab.h"
#include "THGenerateAllTypes.h"

#include "generic/THLabConv.h"
#include "THGenerateAllTypes.h"
