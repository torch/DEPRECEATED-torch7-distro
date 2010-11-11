#
# Find the native cairo includes and library
#
# CAIRO_FOUND -> yes or no if it is found
# CAIRO_LIBRARIES -> full path list of required libraries
# CAIRO_<xxx>_LIBRARY -> path to <xxx> library. Note that <xxx> includes "cairo".
# CAIRO_INCLUDE_DIR -> list of include directories
#

INCLUDE(FindPkgConfig)

PKG_SEARCH_MODULE(_CAIRO libcairo>=1.0.0 cairo>=1.0.0)
SET(CAIRO_FOUND ${_CAIRO_FOUND})

IF(_CAIRO_FOUND)
  SET(CAIRO_LIBRARIES)
  FOREACH(_lib ${_CAIRO_LIBRARIES})
    FIND_LIBRARY(CAIRO_${_lib}_LIBRARY NAMES ${_lib} PATHS ${_CAIRO_LIBRARY_DIR} NO_DEFAULT_PATH)
    FIND_LIBRARY(CAIRO_${_lib}_LIBRARY NAMES ${_lib})
    MARK_AS_ADVANCED(CAIRO_${_lib}_LIBRARY)
    SET(CAIRO_LIBRARIES ${CAIRO_LIBRARIES} ${CAIRO_${_lib}_LIBRARY})
  ENDFOREACH(_lib)
  SET(CAIRO_INCLUDE_DIR ${_CAIRO_INCLUDE_DIRS})
  SET(CAIRO_FOUND 1)
ENDIF(_CAIRO_FOUND)

IF (NOT CAIRO_FOUND AND Cairo_FIND_REQUIRED)
   MESSAGE(FATAL_ERROR "Could not find Cairo -- please install the tools")
ENDIF (NOT CAIRO_FOUND AND Cairo_FIND_REQUIRED)

IF(NOT Cairo_FIND_QUIETLY)
  IF(CAIRO_FOUND)
    MESSAGE(STATUS "Cairo library found")
  ELSE(CAIRO_FOUND)
    MESSAGE(STATUS "Cairo library not found. Please specify library location")
  ENDIF(CAIRO_FOUND)
ENDIF(NOT Cairo_FIND_QUIETLY)
