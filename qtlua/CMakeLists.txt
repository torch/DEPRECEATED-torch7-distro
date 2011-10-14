# -*- cmake -*-



# --- General

SET(WITH_QTLUA ON 
  CACHE BOOL "Compile qlua and associated packages")

SET(WITH_QTLUA_IDE ON
  CACHE BOOL "Compile qlua ide (require Qt>=4.4)")

# --- Macros

IF (WITH_QTLUA)
  SET(CMAKE_QTLUA_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    CACHE INTERNAL "Path for QtLua cmake modules")
  SET(CMAKE_MODULE_PATH ${CMAKE_QTLUA_MODULE_PATH} ${CMAKE_MODULE_PATH})
  INCLUDE(MacroQt4AutoGen)
  INCLUDE(MacroAddQtLuaModule)
  INCLUDE(CheckCXXSourceCompiles)
  INCLUDE(CheckCXXSourceRuns)
ENDIF (WITH_QTLUA)


# --- Qt4

IF (WITH_QTLUA AND NOT QT4_FOUND)
  SET(QT_MIN_VERSION "4.3.0")
  FIND_PACKAGE(Qt4)
  IF (NOT QT4_FOUND)
    MESSAGE(STATUS "Disabling QtLua (No suitable Qt4)")
    SET(WITH_QTLUA FALSE CACHE BOOL "Compile qlua and associated packages" FORCE)
  ENDIF (NOT QT4_FOUND)
ENDIF (WITH_QTLUA AND NOT QT4_FOUND)


# --- Lua

# find lua
IF (WITH_QTLUA)
  FIND_PACKAGE(Lua REQUIRED)
  INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})
ENDIF(WITH_QTLUA)

# was lua compiled with c++?
IF (WITH_QTLUA AND NOT OWN_LUA)
  SET(CMAKE_REQUIRED_DEFINITIONS ${LUA_DEFINITIONS})
  SET(CMAKE_REQUIRED_INCLUDES ${LUA_INCLUDE_DIR})
  SET(CMAKE_REQUIRED_LIBRARIES ${LUA_LIBRARIES})
  CHECK_CXX_SOURCE_RUNS("
#include <stdlib.h>
#include \"lua.h\"
#include \"lauxlib.h\"
int result = 1;
int foo(lua_State *L) { 
  try { luaL_error(L,\"error\"); } catch(...) { result=0; throw; } 
  return 0; }
int main() { lua_State *L = luaL_newstate();
  lua_pushcfunction(L, foo);
  lua_pcall(L, 0, 0, 0); exit(result); }" 
  LUA_COMPILED_WITH_CXX )
  SET(CMAKE_REQUIRED_DEFINITIONS)
  SET(CMAKE_REQUIRED_INCLUDES)
  SET(CMAKE_REQUIRED_LIBRARIES)
  MESSAGE(STATUS "LUA_COMPILED_WITH_CXX=${LUA_COMPILED_WITH_CXX}")
  IF (NOT "${LUA_COMPILED_WITH_CXX}" EQUAL "0")
    MESSAGE(STATUS "Building QtLua against non-C++ Lua")
    #SET(WITH_QTLUA FALSE)
    SET(CMAKE_CXX_FLAGS "-DLUA_NOT_CXX")
  ENDIF (NOT "${LUA_COMPILED_WITH_CXX}" EQUAL "0")
ENDIF(WITH_QTLUA AND NOT OWN_LUA)



# --- compile libqtlua


IF (WITH_QTLUA)
  # ADD_SUBDIRECTORY(qtutil)
  ADD_SUBDIRECTORY(qtlua)
  FIND_PACKAGE(QtLua REQUIRED)
  ADD_SUBDIRECTORY(qlua)
  ADD_SUBDIRECTORY(packages)
ENDIF (WITH_QTLUA)



# --- install qt libraries

IF (WIN32 AND NOT CYGWIN)
  SET(QT_INSTALL_LIBRARIES "OFF"
    CACHE BOOL "Copy the required Qt DLLs into the Torch install dirs")
ENDIF (WIN32 AND NOT CYGWIN)

IF (WITH_QTLUA AND QT_INSTALL_LIBRARIES)
  # -- turn on all the qt modules we use
  SET(QT_USE_QTUITOOLS 1)
  SET(QT_USE_QTWEBKIT 1)
  SET(QT_USE_QTOPENGL 1)
  SET(QT_USE_QTXML 1)
  SET(QT_USE_QTSVG 1)
  # -- this takes care of the install
  INCLUDE("InstallRequiredQtLibraries")
ENDIF (WITH_QTLUA AND QT_INSTALL_LIBRARIES)
