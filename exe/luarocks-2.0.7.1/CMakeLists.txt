PROJECT(LuaRocks)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# If you want to include Lua in a main project,
# you might want to define those variables yourself
IF(NOT LuaRocks_IS_SUBPROJECT)
  INCLUDE(LuaRocksPaths)
ENDIF()

# system wide config file
CONFIGURE_FILE(cmake/config.lua.in ${CMAKE_CURRENT_BINARY_DIR}/config.lua)
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/config.lua" DESTINATION "${LuaRocks_INSTALL_CONFIG_SUBDIR}")

# luarocks module
INSTALL(DIRECTORY "src/luarocks" DESTINATION "${LuaRocks_INSTALL_LUA_PATH_SUBDIR}")

# binary
