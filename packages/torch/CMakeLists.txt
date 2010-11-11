SET(hdr DiskFile.h File.h Storage.h Tensor.h Timer.h)
SET(src DiskFile.c File.c MemoryFile.c PipeFile.c Storage.c Tensor.c TensorMath.c Timer.c utils.c init.c)
SET(luasrc init.lua File.lua Tensor.lua)

SET(src ${src} ${hdr})
  
ADD_TORCH_PACKAGE(torch "${src}" "${luasrc}" "Basics")

TARGET_LINK_LIBRARIES(torch luaT TH)
