# - Try to find Freetype2
# Once done this will define
#
#  FREETYPE2_FOUND - system has Freetype2
#  FREETYPE2_INCLUDE_DIRS - the Freetype2 include directory
#  FREETYPE2_LIBRARIES - Link these to use Freetype2
#  FREETYPE2_DEFINITIONS - Compiler switches required for using Freetype2
#
#  Copyright (c) 2007 Andreas Schneider <mail@cynapses.org>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#


if (FREETYPE2_LIBRARIES AND FREETYPE2_INCLUDE_DIRS)
  # in cache already
  set(FREETYPE2_FOUND TRUE)
else (FREETYPE2_LIBRARIES AND FREETYPE2_INCLUDE_DIRS)
  # use pkg-config to get the directories and then use these values
  # in the FIND_PATH() and FIND_LIBRARY() calls
  include(UsePkgConfig)

  pkgconfig(freetype2 _Freetype2IncDir _Freetype2LinkDir _Freetype2LinkFlags _Freetype2Cflags)

  set(FREETYPE2_DEFINITIONS ${_Freetype2Cflags})

  find_path(FREETYPE2_INCLUDE_DIR
    NAMES
      freetype/freetype.h
    PATHS
      ${_Freetype2IncDir}
      /usr/include
      /usr/local/include
      /opt/local/include
      /sw/include
    PATH_SUFFIXES
      freetype2
  )

  find_library(FREETYPE_LIBRARY
    NAMES
      freetype
    PATHS
      ${_Freetype2LinkDir}
      /usr/lib
      /usr/local/lib
      /opt/local/lib
      /sw/lib
  )

  if (FREETYPE_LIBRARY)
    set(FREETYPE_FOUND TRUE)
  endif (FREETYPE_LIBRARY)

  set(FREETYPE2_INCLUDE_DIRS
    ${FREETYPE2_INCLUDE_DIR}
  )

  if (FREETYPE_FOUND)
    set(FREETYPE2_LIBRARIES
      ${FREETYPE2_LIBRARIES}
      ${FREETYPE_LIBRARY}
    )
  endif (FREETYPE_FOUND)

  if (FREETYPE2_INCLUDE_DIRS AND FREETYPE2_LIBRARIES)
     set(FREETYPE2_FOUND TRUE)
  endif (FREETYPE2_INCLUDE_DIRS AND FREETYPE2_LIBRARIES)

  if (FREETYPE2_FOUND)
    if (NOT Freetype2_FIND_QUIETLY)
      message(STATUS "Found Freetype2: ${FREETYPE2_LIBRARIES}")
    endif (NOT Freetype2_FIND_QUIETLY)
  else (FREETYPE2_FOUND)
    if (Freetype2_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find Freetype2")
    endif (Freetype2_FIND_REQUIRED)
  endif (FREETYPE2_FOUND)

  # show the FREETYPE2_INCLUDE_DIRS and FREETYPE2_LIBRARIES variables only in the advanced view
  mark_as_advanced(FREETYPE2_INCLUDE_DIRS FREETYPE2_LIBRARIES)
  mark_as_advanced(FREETYPE2_INCLUDE_DIR FREETYPE_LIBRARY)

endif (FREETYPE2_LIBRARIES AND FREETYPE2_INCLUDE_DIRS)

