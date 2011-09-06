# -*- cmake -*-

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

SET(hdr 
  THGeneral.h THStorage.h THTensor.h THTensorApply.h 
  THBlas.h THTensorMath.h THLogAdd.h THRandom.h THVector.h)
SET(src 
  THGeneral.c THStorage.c THTensor.c THBlas.c
  THTensorMath.c THLogAdd.c THRandom.c
  THTensorRandom.c
  THLab.c
  THFile.c THDiskFile.c THMemoryFile.c)

SET(src ${src} ${hdr})

IF(UNIX)
  INCLUDE(CheckFunctionExists)
  SET(CMAKE_EXTRA_INCLUDE_FILES "sys/mman.h")
  CHECK_FUNCTION_EXISTS(mmap HAVE_MMAP)
  IF(HAVE_MMAP)
    ADD_DEFINITIONS(-DHAVE_MMAP=1)
  ENDIF(HAVE_MMAP)
ENDIF(UNIX)

ADD_LIBRARY(TH SHARED ${src})

FIND_PACKAGE(BLAS)

IF(BLAS_FOUND)
  ADD_DEFINITIONS(-DUSE_LAPACK)
#  INCLUDE_DIRECTORIES(${CBLAS_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(TH ${BLAS_LIBRARIES})
ENDIF(BLAS_FOUND)

#CONFIGURE_FILE("THCBlas.h.in" "${CMAKE_CURRENT_BINARY_DIR}/THCBlas.h")
#INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
#INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/THCBlas.h" 
#  DESTINATION "${Torch_INSTALL_INCLUDE_SUBDIR}/TH")

INSTALL(TARGETS TH
          RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
          LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
          ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")

INSTALL(FILES
          TH.h
          THBlas.h
          THVector.h
          THLab.h
          THGeneral.h
          THLogAdd.h
          THRandom.h
          THStorage.h
          THTensorMacros.h
          THTensor.h
          THTensorApply.h
          THTensorDimApply.h
          THGenerateAllTypes.h
          THGenerateFloatTypes.h
          THGenerateIntTypes.h
          THFile.h
          THDiskFile.h
          THFilePrivate.h
          THMemoryFile.h
          THTensorMath.h
          THTensorRandom.h
        DESTINATION "${Torch_INSTALL_INCLUDE_SUBDIR}/TH")

INSTALL(FILES
          generic/THBlas.h
          generic/THLab.h
          generic/THLabConv.h
          generic/THStorage.h
          generic/THTensor.h
          generic/THTensorMath.h
          generic/THTensorRandom.h
          generic/THBlas.c
          generic/THStorageCopy.h
          generic/THTensorCopy.h
          generic/THLab.c
          generic/THVector.c
          generic/THStorage.c
          generic/THTensor.c
          generic/THTensorMath.c
          generic/THTensorRandom.c
        DESTINATION "${Torch_INSTALL_INCLUDE_SUBDIR}/TH/generic")

# Create THConfig.cmake
GET_TARGET_PROPERTY(TH_OUTPUT_NAME TH LOCATION)
GET_FILENAME_COMPONENT(TH_OUTPUT_NAME ${TH_OUTPUT_NAME} NAME)
SET(TH_LIBRARIES "${Torch_INSTALL_LIB}/${TH_OUTPUT_NAME}")
SET(TH_INCLUDE_DIR "${Torch_INSTALL_INCLUDE}/TH")
CONFIGURE_FILE(THConfig.cmake.in "${Torch_BINARY_DIR}/cmake-external/THConfig.cmake")
INSTALL(FILES "${Torch_BINARY_DIR}/cmake-external/THConfig.cmake" 
  DESTINATION "${Torch_INSTALL_CMAKE_SUBDIR}")
