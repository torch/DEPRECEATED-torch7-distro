SET(src Storage.c init.c Tensor.c TensorMath.c)
SET(luasrc init.lua Tensor.lua)

SET(src ${src})

INCLUDE_DIRECTORIES("${Torch_SOURCE_DIR}/packages/torch")
  
ADD_TORCH_PACKAGE(cutorch "${src}" "${luasrc}")
TARGET_LINK_LIBRARIES(cutorch luaT THC)
