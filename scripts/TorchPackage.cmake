# -*- cmake -*-
# TORCH_PACKAGE(package package_dependency1 package_dependency2 ...)
# Build a Torch package which depends on the given dependencies.
# (The dependencies are only C/C++ library dependencies)
#
# Variables used: INCLUDE_DIRECTORIES, DEFINITIONS
# Special: - if SWIG_NO_FACTORY is defined, SWIG will not generate
#            factories related methods proper to Torch
#          - if TORCH_PACKAGE_VERBOSE is defined, some extra messages
#            will be displayed
#
# What it does:
#   1. Check for SWIG input file ${package}.i. If it exists,
#      generate the SWIG wrapper.
#
#   2. Compiles C/C++ sources if existing. Combine with SWIG
#      wrapper to generate a shared library  [target ${package}].
#
#   3. Add an entry in the main help index.hlp if the file help/index.hlp
#      exists for this package. [target ${package}-help]
#
#   3. At installation time, install
#       a. the library in lib/
#       b. all *.lua files in lua/
#       c. all help/*.hlp files in help/
#
#

FIND_PACKAGE(Lua REQUIRED)

MACRO(ADD_TORCH_PACKAGE package src luasrc)

  INCLUDE_DIRECTORIES(BEFORE ${LUA_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

  ### C/C++ sources
  IF(src)      

    ADD_LIBRARY(${package} SHARED ${src})
    
    ### Torch packages supposes libraries prefix is "lib"
    SET_TARGET_PROPERTIES(${package} PROPERTIES
      PREFIX "lib"
      IMPORT_PREFIX "lib"
      INSTALL_NAME_DIR "@executable_path/${Torch_INSTALL_BIN2CPATH}")
    
    INSTALL(TARGETS ${package} 
      RUNTIME DESTINATION ${Torch_INSTALL_LUA_CPATH_SUBDIR}
      LIBRARY DESTINATION ${Torch_INSTALL_LUA_CPATH_SUBDIR})
    
  ELSE(src)
    
    IF(TORCH_PACKAGE_VERBOSE)
      MESSAGE(STATUS "Package <${package}>: no C/C++ sources found for package")
    ENDIF(TORCH_PACKAGE_VERBOSE)
    
  ENDIF(src)
  
  ### lua sources
  IF(luasrc)
    INSTALL(FILES ${luasrc} 
      DESTINATION ${Torch_INSTALL_LUA_PATH_SUBDIR}/${package})
  ENDIF(luasrc)
  
  ### help sources
  ADD_TORCH_HELP(${package} ${ARGN})
  
ENDMACRO(ADD_TORCH_PACKAGE)
