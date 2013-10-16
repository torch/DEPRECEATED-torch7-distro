IF("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c89 -pedantic")
ENDIF()

# We compile all existing packages
FILE(GLOB _all_files *)
FOREACH(_file ${_all_files})
  IF(EXISTS "${_file}/CMakeLists.txt")
    ADD_SUBDIRECTORY(${_file})
  ENDIF(EXISTS "${_file}/CMakeLists.txt")
ENDFOREACH(_file ${_all_files})
