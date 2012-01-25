FIND_PACKAGE(CUDA 4.0)

IF (CUDA_FOUND)

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

ELSE (CUDA_FOUND)

    MESSAGE(STATUS "Disabling CUDA (CUDA 4.0 required, and not found)")
    MESSAGE(STATUS "If CUDA 4.0 is install, then specify CUDA_TOOLKIT_ROOT_DIR")

ENDIF (CUDA_FOUND)
