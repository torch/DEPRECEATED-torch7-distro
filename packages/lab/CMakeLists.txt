SET(src lab.c utils.c init.c)
SET(luasrc init.lua hist.lua)
SET(luasrc ${luasrc} test/test.lua)

# Necessary do generate wrapper
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
MACRO(ADD_TORCH_WRAP target luafile)
  GET_FILENAME_COMPONENT(_file_ "${luafile}" NAME_WE)
  SET(cfile "${_file_}.c")
  ADD_CUSTOM_COMMAND(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${cfile}"
    COMMAND "${LUA_EXECUTABLE}" ARGS "-e \"dofile('${CMAKE_SOURCE_DIR}/packages/wrap/init.lua')\" " "${CMAKE_CURRENT_SOURCE_DIR}/${luafile}" "${CMAKE_CURRENT_BINARY_DIR}/${cfile}"
    DEPENDS "${luafile}")
  ADD_CUSTOM_TARGET(${target} DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${cfile}")
ENDMACRO(ADD_TORCH_WRAP)

ADD_TORCH_WRAP(labwrap labwrap.lua)

ADD_TORCH_PACKAGE(lab "${src}" "${luasrc}" "Lab")
ADD_TORCH_DOK(dok lab "Fundamentals" "Matlab-like operations" 1.2)

TARGET_LINK_LIBRARIES(lab luaT TH)
ADD_DEPENDENCIES(lab labwrap)
