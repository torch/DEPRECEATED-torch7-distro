# We want release compilation by default
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# When using gcc
IF(CMAKE_COMPILER_IS_GNUCC)
  # we want warnings
  ADD_DEFINITIONS("-Wall -Wno-unused")
  # we want exceptions support even when compiling c code
  ADD_DEFINITIONS("-fexceptions")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

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
