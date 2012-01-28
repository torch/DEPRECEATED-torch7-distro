PROJECT(LuaRocks)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# If you want to include Lua in a main project,
# you might want to define those variables yourself
IF(NOT LuaRocks_IS_SUBPROJECT)
  INCLUDE(LuaRocksInitPaths)
  FIND_PACKAGE(Lua REQUIRED)
  SET(LuaRocks_LUA_EXECUTABLE "${LUA_EXECUTABLE}")
ENDIF()
INCLUDE(LuaRocksPaths)

# system wide config file
CONFIGURE_FILE(cmake/config.lua.in ${CMAKE_CURRENT_BINARY_DIR}/config.lua)
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/config.lua" DESTINATION "${LuaRocks_INSTALL_CONFIG_SUBDIR}")

# site config file
FIND_PROGRAM(LuaRocks_DOWNLOADER_ NAMES wget curl DOC "File downloader (wget or curl)")
IF(LuaRocks_DOWNLOADER)
  GET_FILENAME_COMPONENT(LuaRocks_DOWNLOADER "${LuaRocks_DOWNLOADER_}" NAME)  
ELSE(LuaRocks_DOWNLOADER)
  MESSAGE(FATAL_ERROR "Downloader (wget, curl or fetch) not found")  
ENDIF(LuaRocks_DOWNLOADER)

FIND_PROGRAM(LuaRocks_MD5CHECKER NAMES md5 md5sum openssl DOC "MD5 checker")
IF(NOT LuaRocks_MD5CHECKER)
  MESSAGE(FATAL_ERROR "MD5 checker (md5sum or openssl) not found")
ENDIF(NOT LuaRocks_MD5CHECKER)

EXEC_PROGRAM(uname ARGS -m OUTPUT_VARIABLE LuaRocks_CNAME_M)

CONFIGURE_FILE(cmake/site_config.lua.in ${CMAKE_CURRENT_BINARY_DIR}/site_config.lua)
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/site_config.lua" DESTINATION "${LuaRocks_INSTALL_LUA_PATH_SUBDIR}/luarocks")

# luarocks module
INSTALL(DIRECTORY "src/luarocks" DESTINATION "${LuaRocks_INSTALL_LUA_PATH_SUBDIR}")

# binary
CONFIGURE_FILE(src/bin/luarocks.in ${CMAKE_CURRENT_BINARY_DIR}/${LuaRocks_NAME})
CONFIGURE_FILE(src/bin/luarocks-admin.in ${CMAKE_CURRENT_BINARY_DIR}/${LuaRocks_NAME}-admin)
INSTALL(PROGRAMS
  ${CMAKE_CURRENT_BINARY_DIR}/${LuaRocks_NAME}
  ${CMAKE_CURRENT_BINARY_DIR}/${LuaRocks_NAME}-admin
  DESTINATION "${LuaRocks_INSTALL_BIN}")
