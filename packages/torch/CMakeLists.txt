SET(src DiskFile.c File.c MemoryFile.c PipeFile.c Storage.c Tensor.c TensorMath.c Timer.c utils.c init.c)
SET(luasrc init.lua File.lua Tensor.lua CmdLine.lua Tester.lua)

SET(src ${src})
  
ADD_TORCH_PACKAGE(torch "${src}" "${luasrc}" "Basics")
ADD_TORCH_DOK(torch "Fundamentals" "torch package" 1)

TARGET_LINK_LIBRARIES(torch luaT TH)

CONFIGURE_FILE(torch.in "${Torch_BINARY_DIR}/torch")
INSTALL(FILES "${Torch_BINARY_DIR}/torch"
        DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_EXECUTE GROUP_READ
                    WORLD_EXECUTE WORLD_READ)
