# This CMakeLists.txt has been first taken from LuaDist
# Copyright (C) 2007-2011 LuaDist.
# Created by Peter Drahoš
# Redistribution and use of this file is allowed according to the terms of the MIT license.
# Debugged and modified by Ronan Collobert, for Torch

project(Lua C ASM)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

set ( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH} )

IF(NOT Lua_INSTALL_BIN_SUBDIR)
  SET(Lua_INSTALL_BIN_SUBDIR "bin" CACHE PATH "Install dir for binaries")
  SET(Lua_INSTALL_LIB_SUBDIR "lib" CACHE PATH "Install dir for libraries")
  SET(Lua_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "Install dir for includes")
  SET(Lua_INSTALL_LUA_PATH_SUBDIR "share/lua/5.1" CACHE PATH "Install dir for Lua packages files")
  SET(Lua_INSTALL_LUA_CPATH_SUBDIR "lib/lua/5.1" CACHE PATH "Install dir for Lua C packages files")
ENDIF()
IF(NOT Lua_INSTALL_BIN_RIDBUS)
  FILE(RELATIVE_PATH Lua_INSTALL_BIN_RIDBUS "${Lua_INSTALL_PREFIX}/${Lua_INSTALL_BIN}" "${Lua_INSTALL_PREFIX}/.")
ENDIF()
IF(NOT Lua_INSTALL_FINDLUA_DIR)
  SET(Lua_INSTALL_FINDLUA_DIR "{Lua_BINARY_DIR}/cmake")
ENDIF()

INCLUDE(CheckLibraryExists)

# LuaJIT specific
option ( LUAJIT_DISABLE_FFI "Disable FFI." OFF )
option ( LUAJIT_ENABLE_LUA52COMPAT "Enable Lua 5.2 compatibility." ON )
option ( LUAJIT_DISABLE_JIT "Disable JIT." OFF )
option ( LUAJIT_CPU_SSE2 "Disable SSE2." OFF )
option ( LUAJIT_CPU_NOCMOV "Disable NOCMOV." OFF )
option ( LUAJIT_RUN_DYNASM "Re-generate headers with DynASM." OFF )

if ( WIN32 AND NOT CYGWIN )
  add_definitions ( -DLUA_BUILD_AS_DLL -DLUAJIT_OS=LUAJIT_OS_WINDOWS)
  set ( LJVM_MODE coffasm )
elseif ( APPLE )
  set ( CMAKE_EXE_LINKER_FLAGS "-pagezero_size 10000 -image_base 100000000 ${CMAKE_EXE_LINKER_FLAGS}" )
  set ( LJVM_MODE machasm )
else ()
  set ( LJVM_MODE elfasm )  
endif ()

IF(NOT WIN32)
  CHECK_LIBRARY_EXISTS(dl dlopen "" LUA_USE_DLOPEN)
  IF(NOT LUA_USE_DLOPEN)
    MESSAGE(FATAL_ERROR "Cannot compile a useful lua.
Function dlopen() seems not to be supported on your platform.
Apparently you are not on a Windows platform as well.
So lua has no way to deal with shared libraries!")
  ENDIF(NOT LUA_USE_DLOPEN)
  list ( APPEND LIBS dl )
ENDIF(NOT WIN32)

CHECK_LIBRARY_EXISTS(m sin "" LUA_USE_LIBM)
if ( LUA_USE_LIBM )
  list ( APPEND LIBS m )
endif ()

FIND_PACKAGE(Readline)
IF(READLINE_FOUND)
  INCLUDE_DIRECTORIES(${READLINE_INCLUDE_DIR})
  list(APPEND LIBS ${READLINE_LIBRARIES})
  SET(LUAJIT_USE_READLINE 1)
ENDIF(READLINE_FOUND)

if ( LUA_WIN )
  # Add extra rc files to the windows build
  if ( MSVC OR MINGW )
    set ( LUA_DEF src/lua.def )
    set ( LUA_DLL_RC src/lua_dll.rc )
    set ( LUA_RC src/lua.rc )
    set ( LUAC_RC src/luac.rc )
  endif ()
endif ()

IF (CMAKE_SHARED_LIBRARY_SUFFIX STREQUAL CMAKE_SHARED_MODULE_SUFFIX)
  SET(LUA_USE_MODULE_AND_LIBRARY 0)
ELSE (CMAKE_SHARED_LIBRARY_SUFFIX STREQUAL CMAKE_SHARED_MODULE_SUFFIX)
  SET(LUA_USE_MODULE_AND_LIBRARY 1)
ENDIF (CMAKE_SHARED_LIBRARY_SUFFIX STREQUAL CMAKE_SHARED_MODULE_SUFFIX)

## SOURCES
# Generate luaconf.h
configure_file ( src/luaconf.h.in ${CMAKE_CURRENT_BINARY_DIR}/luaconf.h )

SET(SRC_LJLIB src/lib_base.c src/lib_math.c src/lib_bit.c src/lib_string.c src/lib_table.c
  src/lib_io.c src/lib_os.c src/lib_package.c src/lib_debug.c src/lib_jit.c src/lib_ffi.c)

SET(SRC_LJCORE src/lj_gc.c src/lj_err.c src/lj_char.c src/lj_bc.c src/lj_obj.c
  src/lj_str.c src/lj_tab.c src/lj_func.c src/lj_udata.c src/lj_meta.c src/lj_debug.c
  src/lj_state.c src/lj_dispatch.c src/lj_vmevent.c src/lj_vmmath.c src/lj_api.c
  src/lj_lex.c src/lj_parse.c src/lj_bcread.c src/lj_bcwrite.c
  src/lj_ir.c src/lj_opt_mem.c src/lj_opt_fold.c src/lj_opt_narrow.c
  src/lj_opt_dce.c src/lj_opt_loop.c #only included src/lj_opt_split.c
  src/lj_mcode.c src/lj_snap.c src/lj_record.c src/lj_crecord.c src/lj_ffrecord.c
  src/lj_asm.c src/lj_trace.c #only included src/lj_gdbjit.c
  src/lj_ctype.c src/lj_cdata.c src/lj_cconv.c src/lj_ccall.c src/lj_ccallback.c
  src/lj_carith.c src/lj_clib.c src/lj_cparse.c
  src/lj_lib.c src/lj_alloc.c src/lib_aux.c
  ${SRC_LJLIB} src/lib_init.c src/luaconf.c)

SET(SRC_BUILDVM src/buildvm.c src/buildvm_asm.c src/buildvm_peobj.c src/buildvm_lib.c src/buildvm_fold.c)

# Optionally rebuild buildvm_*.h header.
if (LUAJIT_RUN_DYNASM)
  set(DASM ${LUA} ${CMAKE_CURRENT_SOURCE_DIR}/dynasm/dynasm.lua)
  set(DASM_FLAGS_X86)
  set(DASM_FLAGS_X64 -D X64)
  set(DASM_FLAGS_X64WIN -D X64 -D X64WIN)
  set(DASM_FLAGS_ARM)
  set(DASM_FLAGS_PPCSPE -D SPE)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/buildvm_x86.h
    COMMAND ${DASM} -LN ${DASM_FLAGS_X86} -o ${CMAKE_CURRENT_BINARY_DIR}/buildvm_x86.h ${CMAKE_CURRENT_SOURCE_DIR}/src/buildvm_x86.dasc
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dynasm/dynasm.lua
  )
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/buildvm_x64.h
    COMMAND ${DASM} -LN ${DASM_FLAGS_X64} -o ${CMAKE_CURRENT_BINARY_DIR}/buildvm_x64.h ${CMAKE_CURRENT_SOURCE_DIR}/src/buildvm_x86.dasc
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dynasm/dynasm.lua
  )
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/buildvm_x64win.h
    COMMAND ${DASM} -LN ${DASM_FLAGS_X64WIN} -o ${CMAKE_CURRENT_BINARY_DIR}/buildvm_x64win.h ${CMAKE_CURRENT_SOURCE_DIR}/src/buildvm_x86.dasc
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dynasm/dynasm.lua
  )
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/buildvm_arm.h
    COMMAND ${DASM} -LN ${DASM_FLAGS_PPCSPE} -o ${CMAKE_CURRENT_BINARY_DIR}/buildvm_arm.h ${CMAKE_CURRENT_SOURCE_DIR}/src/buildvm_arm.dasc
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dynasm/dynasm.lua
  )
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/buildvm_ppcspe.h
    COMMAND ${DASM} -LN ${DASM_FLAGS_PPCSPE} -o ${CMAKE_CURRENT_BINARY_DIR}/buildvm_ppcspe.h ${CMAKE_CURRENT_SOURCE_DIR}/src/buildvm_ppc.dasc
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dynasm/dynasm.lua
  )
  
  # Note: not all of these are needed for the current platform, but for simplicity
  # specify all.
  set_source_files_properties(src/buildvm.c PROPERTIES OBJECT_DEPENDS
	${CMAKE_CURRENT_BINARY_DIR}/buildvm_x86.h
	${CMAKE_CURRENT_BINARY_DIR}/buildvm_x64.h
	${CMAKE_CURRENT_BINARY_DIR}/buildvm_x64win.h
	${CMAKE_CURRENT_BINARY_DIR}/buildvm_arm.h
	${CMAKE_CURRENT_BINARY_DIR}/buildvm_ppcspe.h)
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
endif ()

## GENERATE
add_executable ( buildvm ${SRC_BUILDVM} )

macro(add_buildvm_target _target _mode)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_target}
    COMMAND buildvm ARGS -m ${_mode} -o ${CMAKE_CURRENT_BINARY_DIR}/${_target} ${ARGN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS buildvm ${ARGN}
  )
endmacro(add_buildvm_target)

if (MSVC)
  add_buildvm_target ( lj_vm.obj peobj )
  set (LJ_VM_SRC ${CMAKE_CURRENT_BINARY_DIR}/lj_vm.obj)
else ()
  add_buildvm_target ( lj_vm.s ${LJVM_MODE} )
  set (LJ_VM_SRC ${CMAKE_CURRENT_BINARY_DIR}/lj_vm.s)
endif ()
add_buildvm_target ( lj_ffdef.h   ffdef   ${SRC_LJLIB} )
add_buildvm_target ( lj_bcdef.h  bcdef  ${SRC_LJLIB} )
add_buildvm_target ( lj_folddef.h folddef src/lj_opt_fold.c )
add_buildvm_target ( lj_recdef.h  recdef  ${SRC_LJLIB} )
add_buildvm_target ( lj_libdef.h  libdef  ${SRC_LJLIB} )

SET(DEPS 
  ${LJ_VM_SRC}
  ${CMAKE_CURRENT_BINARY_DIR}/lj_ffdef.h
  ${CMAKE_CURRENT_BINARY_DIR}/lj_bcdef.h
  ${CMAKE_CURRENT_BINARY_DIR}/lj_libdef.h
  ${CMAKE_CURRENT_BINARY_DIR}/lj_recdef.h
  ${CMAKE_CURRENT_BINARY_DIR}/lj_folddef.h 
  )
                  
## COMPILE
include_directories (BEFORE ${CMAKE_CURRENT_BINARY_DIR} dynasm src )

add_library ( liblua-shared SHARED ${SRC_LJCORE} ${DEPS} )
target_link_libraries ( liblua-shared ${LIBS} )
SET_TARGET_PROPERTIES(liblua-shared PROPERTIES 
  PREFIX "lib" IMPORT_PREFIX "lib")

add_library ( liblua-static STATIC ${SRC_LJCORE} ${DEPS} )
target_link_libraries ( liblua-static ${LIBS} )
SET_TARGET_PROPERTIES(liblua-static PROPERTIES
  COMPILE_DEFINITIONS "liblua_STATIC"
  OUTPUT_NAME "lua-static")

add_executable ( lua src/luajit.c)
target_link_libraries ( lua liblua-shared )

add_executable ( lua-static src/luajit.c)
target_link_libraries ( lua-static liblua-static )

INSTALL(TARGETS liblua-shared lua
  RUNTIME DESTINATION "${Lua_INSTALL_BIN_SUBDIR}"
  LIBRARY DESTINATION "${Lua_INSTALL_LIB_SUBDIR}"
  ARCHIVE DESTINATION "${Lua_INSTALL_LIB_SUBDIR}")

# Create internal FindLua.cmake
SET(LUA_LIBRARIES liblua-shared)
SET(LUA_EXECUTABLE lua-static)
SET(LUAC_EXECUTABLE torch-luac)
SET(LUA_INCLUDE_DIR 
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR}
  ${READLINE_INCLUDE_DIR})
CONFIGURE_FILE(cmake/LuaConfig.cmake.in "${Lua_INSTALL_FINDLUA_DIR}/FindLua.cmake")
