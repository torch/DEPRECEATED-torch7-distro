SET(src-cuda init.cu)
SET(luasrc init.lua)

CUDA_ADD_LIBRARY(cunn SHARED ${src-cuda})

TARGET_LINK_LIBRARIES(cunn luaT THC)

### Torch packages supposes libraries prefix is "lib"
SET_TARGET_PROPERTIES(cunn PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib"
  INSTALL_NAME_DIR "@executable_path/${Torch_INSTALL_BIN2CPATH}")
    
INSTALL(TARGETS cunn
  RUNTIME DESTINATION ${Torch_INSTALL_LUA_CPATH_SUBDIR}
  LIBRARY DESTINATION ${Torch_INSTALL_LUA_CPATH_SUBDIR})

ADD_TORCH_PACKAGE(cunn "" "${luasrc}")
