FIND_PACKAGE(CUDA 4.0 REQUIRED)

# bug on Apple
IF(APPLE)
  LINK_DIRECTORIES("/usr/local/cuda/lib/")
ENDIF(APPLE)

FILE(GLOB _all_files *)

FOREACH(_file ${_all_files})
  IF(EXISTS "${_file}/CMakeLists.txt")
    ADD_SUBDIRECTORY(${_file})
  ENDIF(EXISTS "${_file}/CMakeLists.txt")
ENDFOREACH(_file ${_all_files})
