#include "THVector.h"

#if defined(__SSE2__)

#else

// If SSE2 not defined, then generate plain C operators
#include "generic/THVector.c"
#include "THGenerateFloatTypes.h"

#endif

// For non-float types, generate plain C operators
#include "generic/THVector.c"
#include "THGenerateIntTypes.h"
