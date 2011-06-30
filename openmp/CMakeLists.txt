
FIND_PACKAGE(OPENMP REQUIRED)

IF(OPENMP_FOUND)
  MESSAGE(STATUS "OpenMP Found with compiler flag : ${OpenMP_C_FLAGS}")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
ENDIF(OPENMP_FOUND)

FILE(GLOB _all_files *)

FOREACH(_file ${_all_files})
  IF(EXISTS "${_file}/CMakeLists.txt")
    ADD_SUBDIRECTORY(${_file})
  ENDIF(EXISTS "${_file}/CMakeLists.txt")
ENDFOREACH(_file ${_all_files})

