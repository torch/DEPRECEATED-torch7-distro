MACRO(ADD_TORCH_WRAP target luafile)
  INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
  GET_FILENAME_COMPONENT(_file_ "${luafile}" NAME_WE)
  SET(cfile "${_file_}.c")
  ADD_CUSTOM_COMMAND(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${cfile}"
    COMMAND "${LUA_EXECUTABLE}" ARGS "-e \"dofile('${CMAKE_SOURCE_DIR}/pkg/wrap/init.lua')\" " "${CMAKE_CURRENT_SOURCE_DIR}/${luafile}" "${CMAKE_CURRENT_BINARY_DIR}/${cfile}"
    DEPENDS "${luafile}" "${LUA_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/pkg/wrap/init.lua" "${CMAKE_SOURCE_DIR}/pkg/wrap/types.lua")
  ADD_CUSTOM_TARGET(${target} DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${cfile}")
ENDMACRO(ADD_TORCH_WRAP)
