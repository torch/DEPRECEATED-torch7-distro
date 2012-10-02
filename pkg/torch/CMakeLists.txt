SET(src)
# DiskFile.c File.c MemoryFile.c PipeFile.c Timer.c utils.c init.c random.c)
SET(luasrc init.lua File.lua Storage.lua StorageCopy.lua Tensor.lua print.lua CmdLine.lua Tester.lua test/test.lua ${CMAKE_CURRENT_BINARY_DIR}/TensorMath.lua)
  
# Necessary do generate wrapper
ADD_TORCH_WRAP(tensormathwrap TensorMath.lua)
#ADD_TORCH_WRAP(randomwrap random.lua)

ADD_TORCH_PACKAGE(torch "${src}" "${luasrc}")
ADD_TORCH_DOK(dok torch "Fundamentals" "Torch package" 1.1)
ADD_DEPENDENCIES(torch tensormathwrap)

#TARGET_LINK_LIBRARIES(torch luaT TH)

INSTALL(FILES generic/Storage.c generic/Tensor.c
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/torch/generic")
