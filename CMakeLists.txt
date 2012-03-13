# -*- cmake -*-

PROJECT(Torch)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake"
                      "${CMAKE_BINARY_DIR}/cmake" "${CMAKE_MODULE_PATH}")

INCLUDE(TorchPaths)
INCLUDE(TorchPathsInit)
INCLUDE(TorchCFlags)
INCLUDE(TorchDOK)
INCLUDE(TorchLua2exe)
INCLUDE(TorchPackage)
INCLUDE(TorchWrap)

FIND_PACKAGE(OpenMP)

IF(OPENMP_FOUND)
  MESSAGE(STATUS "OpenMP Found with compiler flag : ${OpenMP_C_FLAGS}")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
ENDIF(OPENMP_FOUND)

# Torch executables
ADD_SUBDIRECTORY(exe)

# Torch libraries
ADD_SUBDIRECTORY(lib)

# Torch base packages
ADD_SUBDIRECTORY(pkg)

# Torch extra packages
ADD_SUBDIRECTORY(extra)

# External packages support
INCLUDE(TorchExports)

# Packaging support
INCLUDE(TorchCPack)
