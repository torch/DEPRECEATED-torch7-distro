FIND_PACKAGE(Wget)
FIND_PACKAGE(MD5)

IF(NOT WGET_FOUND)
  MESSAGE(FATAL_ERROR "Downloader wget not found")
ENDIF()

IF(NOT MD5_FOUND)
  MESSAGE(FATAL_ERROR "MD5 checker not found")
ENDIF()

GET_FILENAME_COMPONENT(WGET_EXECUTABLE_NAME "${WGET_EXECUTABLE}" NAME)
GET_FILENAME_COMPONENT(MD5_EXECUTABLE_NAME "${MD5_EXECUTABLE}" NAME)

IF(CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
  EXEC_PROGRAM(uname ARGS -m OUTPUT_VARIABLE UNAME_M RETURN_VALUE val)
  IF("${val}" GREATER 0)
    SET(UNAME_M ${CMAKE_SYSTEM_PROCESSOR})
  ENDIF()
ELSE()
  SET(UNAME_M ${CMAKE_SYSTEM_PROCESSOR})
ENDIF()

CONFIGURE_FILE("src/luarocks/site_config.lua.in" "${CMAKE_CURRENT_BINARY_DIR}/src/luarocks/site_config.lua")
CONFIGURE_FILE("src/luarocks/config.lua.in" "${CMAKE_CURRENT_BINARY_DIR}/src/luarocks/config.lua")
CONFIGURE_FILE("src/bin/luarocks" "${CMAKE_CURRENT_BINARY_DIR}/src/bin/torch-rocks")
CONFIGURE_FILE("src/bin/luarocks-admin" "${CMAKE_CURRENT_BINARY_DIR}/src/bin/torch-rocks-admin")

SET(LUAROCKS_FILES fs/unix/tools.lua fs/unix.lua fs/win32/tools.lua fs/win32.lua
  fs/lua.lua persist.lua list.lua require.lua repos.lua dir.lua make_manifest.lua
  command_line.lua install.lua build/command.lua build/cmake.lua build/make.lua
  build/builtin.lua fetch/cvs.lua fetch/git.lua fetch/sscm.lua tools/patch.lua
  fetch/svn.lua tools/zip.lua tools/tar.lua pack.lua type_check.lua make.lua path.lua
  remove.lua fs.lua manif.lua add.lua deps.lua build.lua search.lua show.lua
  manif_core.lua fetch.lua unpack.lua validate.lua cfg.lua download.lua
  help.lua util.lua index.lua cache.lua refresh_cache.lua loader.lua
  admin_remove.lua fetch/hg.lua fetch/git_file.lua new_version.lua lint.lua purge.lua)

FOREACH(LUAROCKS_FILE ${LUAROCKS_FILES})
  GET_FILENAME_COMPONENT(_subpath "${LUAROCKS_FILE}" PATH)
  INSTALL(FILES "src/luarocks/${LUAROCKS_FILE}" DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/luarocks/${_subpath}")
ENDFOREACH()

INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/src/luarocks/site_config.lua" DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/luarocks")
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/src/luarocks/config.lua" DESTINATION "${Torch_INSTALL_LUAROCKS_SYSCONF_SUBDIR}")
INSTALL(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/src/bin/torch-rocks" DESTINATION "${Torch_INSTALL_BIN_SUBDIR}")
INSTALL(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/src/bin/torch-rocks-admin" DESTINATION "${Torch_INSTALL_BIN_SUBDIR}")

INSTALL(CODE "EXEC_PROGRAM(${Torch_INSTALL_BIN}/torch-rocks ARGS install https://raw.github.com/torch/rocks/master/torch-7.scm-0.rockspec)")
