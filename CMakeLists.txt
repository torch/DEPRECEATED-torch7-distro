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

SET(OMP 1)
IF (APPLE AND CMAKE_COMPILER_IS_GNUCC)
   EXEC_PROGRAM (uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
   STRING (REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
   MESSAGE (STATUS "MAC OS Darwin Version: ${DARWIN_VERSION}")
   IF (DARWIN_VERSION GREATER 10)
      SET(APPLE_LION_SUCKS 1)
   ENDIF (DARWIN_VERSION GREATER 10)
   EXECUTE_PROCESS (COMMAND ${CMAKE_C_COMPILER} -dumpversion
                    OUTPUT_VARIABLE GCC_VERSION)
   IF (APPLE_LION_SUCKS AND GCC_VERSION VERSION_LESS 4.6.2)
      MESSAGE(STATUS "Warning: Disabling OpenMP (unstable with this version of GCC)")
      MESSAGE(STATUS " Install GCC >= 4.6.2 or change your OS to enable OpenMP")
      SET(OMP 0)
   ENDIF ()
ENDIF ()

IF (OMP)
   FIND_PACKAGE(OpenMP)
   IF(OPENMP_FOUND)
      MESSAGE(STATUS "OpenMP Found with compiler flag : ${OpenMP_C_FLAGS}")
      SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
   ENDIF(OPENMP_FOUND)
ENDIF (OMP)

FIND_PACKAGE(ARM)
IF (NEON_FOUND)
  MESSAGE(STATUS "Neon found with compiler flag : -mfpu=neon -D__NEON__")
  SET(CMAKE_C_FLAGS "-mfpu=neon -D__NEON__ ${CMAKE_C_FLAGS}")
ENDIF (NEON_FOUND)
IF (CORTEXA8_FOUND)
  MESSAGE(STATUS "Cortex-A8 Found with compiler flag : -mcpu=cortex-a8")
  SET(CMAKE_C_FLAGS "-mcpu=cortex-a8 -fprefetch-loop-arrays ${CMAKE_C_FLAGS}")
ENDIF (CORTEXA8_FOUND)
IF (CORTEXA9_FOUND)
  MESSAGE(STATUS "Cortex-A9 Found with compiler flag : -mcpu=cortex-a9")
  SET(CMAKE_C_FLAGS "-mcpu=cortex-a9 ${CMAKE_C_FLAGS}")
ENDIF (CORTEXA9_FOUND)

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
