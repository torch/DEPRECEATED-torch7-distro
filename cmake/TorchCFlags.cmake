INCLUDE(CheckCCompilerFlag)
INCLUDE(CheckCXXCompilerFlag)

# We want release compilation by default
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# we want warnings
# we want exceptions support even when compiling c code

# C
CHECK_C_COMPILER_FLAG(-Wall C_HAS_WALL)
IF(C_HAS_WALL)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
ENDIF(C_HAS_WALL)

CHECK_C_COMPILER_FLAG(-Wno-unused-function C_HAS_NO_UNUSED_FUNCTION)
IF(C_HAS_NO_UNUSED_FUNCTION)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
ENDIF(C_HAS_NO_UNUSED_FUNCTION)

CHECK_C_COMPILER_FLAG(-fexceptions C_HAS_FEXCEPTIONS)
IF(C_HAS_FEXCEPTIONS)
  SET(CMAKE_C_FLAGS "-fexceptions ${CMAKE_C_FLAGS}")
ENDIF(C_HAS_FEXCEPTIONS)

# C++
CHECK_CXX_COMPILER_FLAG(-Wall CXX_HAS_WALL)
IF(CXX_HAS_WALL)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
ENDIF(CXX_HAS_WALL)

CHECK_CXX_COMPILER_FLAG(-Wno-unused-function CXX_HAS_NO_UNUSED_FUNCTION)
IF(CXX_HAS_NO_UNUSED_FUNCTION)
  SET(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
ENDIF(CXX_HAS_NO_UNUSED_FUNCTION)

# When using MSVC
IF(MSVC)
  # we want to respect the standard, and we are bored of those **** .
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE=1)
ENDIF(MSVC)

# On Apple
IF(APPLE AND CMAKE_COMPILER_IS_GNUCXX)
  # avoid weird allocation errors [maybe obsolete since libs disctinct from modules]
  SET(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -Wl,-flat_namespace")
  SET(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -Wl,-flat_namespace")
ENDIF(APPLE AND CMAKE_COMPILER_IS_GNUCXX)
