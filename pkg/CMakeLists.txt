# We include all Torch libraries
INCLUDE_DIRECTORIES("${Torch_SOURCE_DIR}/lib/TH")
INCLUDE_DIRECTORIES("${Torch_SOURCE_DIR}/lib/luaT")

# We compile all existing packages
FILE(GLOB _all_files *)
FOREACH(_file ${_all_files})
  IF(EXISTS "${_file}/CMakeLists.txt")
    ADD_SUBDIRECTORY(${_file})
  ENDIF(EXISTS "${_file}/CMakeLists.txt")
ENDFOREACH(_file ${_all_files})
