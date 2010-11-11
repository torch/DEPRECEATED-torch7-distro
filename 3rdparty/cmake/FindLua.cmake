# - Find lua
# this module looks for Lua
#
#  LUA_EXECUTABLE - the full path to lua
#  LUA_LIBRARIES - the lua shared library
#  LUA_INCLUDE_DIR - directory for lua includes
#  LUA_PACKAGE_PATH - where Lua searches for Lua packages
#  LUA_PACKAGE_CPATH - where Lua searches for library packages
#  LUA_FOUND      - If false, don't attempt to use lua.

INCLUDE(FindCygwin)

FIND_PROGRAM(LUA_EXECUTABLE
  lua
  ${CYGWIN_INSTALL_PATH}/bin
  PATH
  )

IF(LUA_EXECUTABLE)
  GET_FILENAME_COMPONENT(LUA_DIR ${LUA_EXECUTABLE} PATH)
ENDIF(LUA_EXECUTABLE)

FIND_LIBRARY(LUA_LIBRARIES
  NAMES lua liblua
  PATHS ${LUA_DIR}/../lib
  ${LUA_DIR}
  NO_DEFAULT_PATH
)

FIND_PATH(LUA_INCLUDE_DIR lua.h
  ${LUA_DIR}/../include
  NO_DEFAULT_PATH
)

SET(LUA_PACKAGE_PATH "${LUA_DIR}/../share/lua/5.1" CACHE PATH "where Lua searches for Lua packages")
SET(LUA_PACKAGE_CPATH "${LUA_DIR}/../lib/lua/5.1" CACHE PATH "where Lua searches for library packages")

MARK_AS_ADVANCED(
  LUA_EXECUTABLE
  LUA_LIBRARIES
  LUA_INCLUDE_DIR
  LUA_PACKAGE_PATH
  LUA_PACKAGE_CPATH
  )

IF(LUA_EXECUTABLE)
  IF(LUA_LIBRARIES)
    IF(LUA_INCLUDE_DIR)
      SET(LUA_FOUND 1)
    ENDIF(LUA_INCLUDE_DIR)
  ENDIF(LUA_LIBRARIES)
ENDIF(LUA_EXECUTABLE)

IF (NOT LUA_FOUND AND Lua_FIND_REQUIRED)
   MESSAGE(FATAL_ERROR "Could not find Lua -- please install the tools")
ENDIF (NOT LUA_FOUND AND Lua_FIND_REQUIRED)

IF(NOT Lua_FIND_QUIETLY)
  IF(LUA_FOUND)
    MESSAGE(STATUS "Lua found")
  ELSE(LUA_FOUND)
    MESSAGE(STATUS "Lua not found. Please specify location")
  ENDIF(LUA_FOUND)
ENDIF(NOT Lua_FIND_QUIETLY)
