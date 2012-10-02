MACRO(ADD_TORCH_WRAP target luafile)
  ADD_CUSTOM_COMMAND(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${luafile}"
    COMMAND ${Torch_SOURCE_LUA} ARGS "-e \"dofile('${Torch_SOURCE_PKG}/wrap/init.lua')\" " "${CMAKE_CURRENT_SOURCE_DIR}/${luafile}" "${CMAKE_CURRENT_BINARY_DIR}/${luafile}"
    DEPENDS "${luafile}" ${Torch_SOURCE_LUA} "${Torch_SOURCE_PKG}/wrap/init.lua" "${Torch_SOURCE_PKG}/wrap/types.lua")
  ADD_CUSTOM_TARGET(${target} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${luafile})
ENDMACRO(ADD_TORCH_WRAP)
