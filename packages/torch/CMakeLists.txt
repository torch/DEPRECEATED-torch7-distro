SET(src DiskFile.c File.c MemoryFile.c PipeFile.c Storage.c Tensor.c TensorMath.c Timer.c utils.c init.c)
SET(luasrc init.lua File.lua Tensor.lua CmdLine.lua Tester.lua)

SET(src ${src})
  
ADD_TORCH_PACKAGE(torch "${src}" "${luasrc}" "Basics")

TARGET_LINK_LIBRARIES(torch luaT TH)
