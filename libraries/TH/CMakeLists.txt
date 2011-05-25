# -*- cmake -*-

SET(hdr 
  THGeneral.h THStorage.h THTensor.h THTensorApply.h 
  THBlas.h THTensorMath.h THLogAdd.h THRandom.h)
SET(src 
  THGeneral.c THStorage.c THTensor.c THBlas.c 
  THTensorMath.c THLogAdd.c THRandom.c
  THTensorRandom.c
  THLab.c)
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

FIND_PACKAGE(CBLAS)

IF(CBLAS_FOUND)
  SET(USE_CBLAS 1)
  INCLUDE_DIRECTORIES(${CBLAS_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(TH ${CBLAS_LIBRARIES})
ENDIF(CBLAS_FOUND)

CONFIGURE_FILE("THCBlas.h.in" "${CMAKE_CURRENT_BINARY_DIR}/THCBlas.h")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/THCBlas.h" 
  DESTINATION "${Torch_INSTALL_INCLUDE_SUBDIR}/TH")

INSTALL(TARGETS TH
          RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
          LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
          ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")

INSTALL(FILES
          TH.h
          THBlas.h
          THGeneral.h
          THLogAdd.h
          THRandom.h
          THStorage.h
          generic/THStorage.h
          THTensor.h
          generic/THTensor.h
          THTensorApply.h
          THTensorDimApply.h
        DESTINATION "${Torch_INSTALL_INCLUDE_SUBDIR}/TH")

# Create THConfig.cmake
GET_TARGET_PROPERTY(TH_OUTPUT_NAME TH LOCATION)
GET_FILENAME_COMPONENT(TH_OUTPUT_NAME ${TH_OUTPUT_NAME} NAME)
SET(TH_LIBRARIES "${Torch_INSTALL_LIB}/${TH_OUTPUT_NAME}")
SET(TH_INCLUDE_DIR "${Torch_INSTALL_INCLUDE}/TH")
CONFIGURE_FILE(THConfig.cmake.in "${Torch_BINARY_DIR}/cmake-external/THConfig.cmake")
INSTALL(FILES "${Torch_BINARY_DIR}/cmake-external/THConfig.cmake" 
  DESTINATION "${Torch_INSTALL_CMAKE_SUBDIR}")
