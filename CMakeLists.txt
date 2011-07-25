# -*- cmake -*-

PROJECT(Torch)

IF (WIN32)
  CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
ELSEIF (APPLE)
  CMAKE_MINIMUM_REQUIRED(VERSION 2.4.8 FATAL_ERROR)
ELSE (WIN32)
  CMAKE_MINIMUM_REQUIRED(VERSION 2.4.7 FATAL_ERROR)
ENDIF(WIN32)

IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(VERSION 2.6)
ENDIF(COMMAND CMAKE_POLICY)

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/scripts 
		      ${CMAKE_BINARY_DIR}/cmake ${CMAKE_MODULE_PATH})

INCLUDE(TorchPaths)

IF(UNIX)
  SET(CMAKE_SKIP_BUILD_RPATH FALSE)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  FILE(RELATIVE_PATH Torch_INSTALL_BIN2LIB 
    "${Torch_INSTALL_BIN}" "${Torch_INSTALL_LIB}")
  FILE(RELATIVE_PATH Torch_INSTALL_BIN2CPATH 
    "${Torch_INSTALL_BIN}" "${Torch_INSTALL_LUA_CPATH}")
  IF(NOT APPLE) 
    OPTION(WITH_DYNAMIC_RPATH 
      "Build libraries with executable relative rpaths (\$ORIGIN)" ON )
  ENDIF(NOT APPLE)
  IF (WITH_DYNAMIC_RPATH OR APPLE)
    SET(CMAKE_INSTALL_RPATH "\$ORIGIN/${Torch_INSTALL_BIN2LIB}")
  ELSE (WITH_DYNAMIC_RPATH OR APPLE)
    SET(CMAKE_INSTALL_RPATH "${Torch_INSTALL_LIB}")
  ENDIF (WITH_DYNAMIC_RPATH OR APPLE)
  SET(CMAKE_INSTALL_NAME_DIR "@executable_path/${Torch_INSTALL_BIN2LIB}")
ENDIF(UNIX)

IF (WIN32)
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
ENDIF (WIN32)

INCLUDE(TorchFlags)

ADD_SUBDIRECTORY(3rdparty)

INCLUDE(TorchHelp)
INCLUDE(TorchPackage)

SET(WEBSITE OFF CACHE BOOLEAN "Produce the Torch website?")
IF(WEBSITE)
  ADD_SUBDIRECTORY(website)
ENDIF(WEBSITE)

ADD_SUBDIRECTORY(libraries)
ADD_SUBDIRECTORY(packages)

ADD_SUBDIRECTORY(qtlua)

# For external packages...
CONFIGURE_FILE(scripts/TorchPaths.cmake.in 
  "${Torch_BINARY_DIR}/cmake-external/TorchPaths.cmake" @ONLY)
INSTALL(FILES "${Torch_BINARY_DIR}/cmake-external/TorchPaths.cmake" 
  DESTINATION "${Torch_INSTALL_CMAKE_SUBDIR}")

INCLUDE(CMakeExportBuildSettings)
CMAKE_EXPORT_BUILD_SETTINGS("${Torch_BINARY_DIR}/cmake-external/TorchBuildSettings.cmake")
INSTALL(FILES "${Torch_BINARY_DIR}/cmake-external/TorchBuildSettings.cmake" DESTINATION "${Torch_INSTALL_CMAKE_SUBDIR}")

CONFIGURE_FILE("${Torch_SOURCE_DIR}/scripts/TorchConfig.cmake.in" "${Torch_BINARY_DIR}/cmake-external/TorchConfig.cmake" COPYONLY)
INSTALL(FILES "${Torch_BINARY_DIR}/cmake-external/TorchConfig.cmake" DESTINATION "${Torch_INSTALL_CMAKE_SUBDIR}")
INSTALL(FILES "${Torch_SOURCE_DIR}/scripts/TorchPackage.cmake" DESTINATION "${Torch_INSTALL_CMAKE_SUBDIR}")

EXPORT_LIBRARY_DEPENDENCIES("${Torch_BINARY_DIR}/cmake-external/TorchLibraryDepends.cmake")
INSTALL(FILES "${Torch_BINARY_DIR}/cmake-external/TorchLibraryDepends.cmake" DESTINATION "${Torch_INSTALL_CMAKE_SUBDIR}")

# Contrib Packages
IF(EXISTS "${Torch_SOURCE_DIR}/contrib/CMakeLists.txt")
  ADD_SUBDIRECTORY(contrib)
ENDIF(EXISTS "${Torch_SOURCE_DIR}/contrib/CMakeLists.txt")

# Milde Packages [NEC internal stuff]
IF(EXISTS "${Torch_SOURCE_DIR}/milde/CMakeLists.txt")
  ADD_SUBDIRECTORY(milde)
ENDIF(EXISTS "${Torch_SOURCE_DIR}/milde/CMakeLists.txt")

# OpenMP
IF(EXISTS "${Torch_SOURCE_DIR}/openmp/CMakeLists.txt")
  ADD_SUBDIRECTORY(openmp)
ENDIF(EXISTS "${Torch_SOURCE_DIR}/openmp/CMakeLists.txt")

# Dev Packages
ADD_SUBDIRECTORY(dev)

# CPack
IF(EXISTS "${Torch_SOURCE_DIR}/milde/CPackLists.txt")
  INCLUDE("${Torch_SOURCE_DIR}/milde/CPackLists.txt")
ELSE(EXISTS "${Torch_SOURCE_DIR}/milde/CPackLists.txt")
  INCLUDE(TorchCPack)
ENDIF(EXISTS "${Torch_SOURCE_DIR}/milde/CPackLists.txt")

