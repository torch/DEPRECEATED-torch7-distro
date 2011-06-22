SET(src Storage.c init.c)
SET(luasrc init.lua)

SET(src ${src})
  
ADD_TORCH_PACKAGE(cutorch "${src}" "${luasrc}" "xxx")
TARGET_LINK_LIBRARIES(cutorch luaT THC)
