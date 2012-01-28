SET(LuaRocks_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

SET(LuaRocks_NAME "luarocks" CACHE PATH
  "Name to be given to luarocks executable or user config file")

SET(LuaRocks_INSTALL_BIN_SUBDIR "bin" CACHE PATH
  "Install dir for binaries (relative to LuaRocks_INSTALL_PREFIX)")

SET(LuaRocks_INSTALL_LIB_SUBDIR "lib" CACHE PATH
  "Install dir for archives (relative to LuaRocks_INSTALL_PREFIX)")

SET(LuaRocks_INSTALL_CONFIG_SUBDIR "etc/luarocks" CACHE PATH
  "Install dir for config file (relative to LuaRocks_INSTALL_PREFIX)")

SET(LuaRocks_INSTALL_SHARE_SUBDIR "share" CACHE PATH
  "Install dir for data (relative to LuaRocks_INSTALL_PREFIX)")

SET(LuaRocks_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH
  "Install dir for include (relative to LuaRocks_INSTALL_PREFIX)")

SET(LuaRocks_INSTALL_CMAKE_SUBDIR "share/luarocks/cmake" CACHE PATH
  "Install dir for .cmake files (relative to LuaRocks_INSTALL_PREFIX)")

SET(LuaRocks_INSTALL_LUA_PATH_SUBDIR "share/lua/5.1" CACHE PATH
  "Install dir for Lua packages files (relative to LuaRocks_INSTALL_PREFIX)")

SET(LuaRocks_INSTALL_LUA_CPATH_SUBDIR "lib/lua/5.1" CACHE PATH
  "Install dir for Lua C packages files (relative to LuaRocks_INSTALL_PREFIX)")
