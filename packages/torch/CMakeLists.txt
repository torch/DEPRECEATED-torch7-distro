SET(src DiskFile.c File.c MemoryFile.c PipeFile.c Storage.c Tensor.c Timer.c utils.c init.c TensorOperator.c TensorMathWrap.c random.c)
SET(luasrc init.lua File.lua Tensor.lua TensorMath.lua CmdLine.lua Tester.lua torch.lua test/test.lua)
  
# Necessary do generate wrapper
MACRO(ADD_TORCH_WRAP target luafile)
  INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
  GET_FILENAME_COMPONENT(_file_ "${luafile}" NAME_WE)
  SET(cfile "${_file_}.c")
  ADD_CUSTOM_COMMAND(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${cfile}"
    COMMAND "${LUA_EXECUTABLE}" ARGS "-e \"dofile('${CMAKE_SOURCE_DIR}/packages/wrap/init.lua')\" " "${CMAKE_CURRENT_SOURCE_DIR}/${luafile}" "${CMAKE_CURRENT_BINARY_DIR}/${cfile}"
    DEPENDS "${luafile}" "${LUA_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/packages/wrap/init.lua" "${CMAKE_SOURCE_DIR}/packages/wrap/types.lua")
  ADD_CUSTOM_TARGET(${target} DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${cfile}")
ENDMACRO(ADD_TORCH_WRAP)

ADD_TORCH_WRAP(tensormathwrap TensorMathWrap.lua)
ADD_TORCH_WRAP(randomwrap random.lua)

ADD_TORCH_PACKAGE(torch "${src}" "${luasrc}" "Basics")
ADD_TORCH_DOK(dok torch "Fundamentals" "Torch package" 1.1)

TARGET_LINK_LIBRARIES(torch luaT TH)

CONFIGURE_FILE(torch.in "${Torch_BINARY_DIR}/torch")
INSTALL(FILES "${Torch_BINARY_DIR}/torch"
        DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_EXECUTE GROUP_READ
                    WORLD_EXECUTE WORLD_READ)
