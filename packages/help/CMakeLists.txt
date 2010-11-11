CONFIGURE_FILE(init.lua.in "${CMAKE_CURRENT_BINARY_DIR}/init.lua")
SET(luasrc "${CMAKE_CURRENT_BINARY_DIR}/init.lua")
ADD_TORCH_PACKAGE(help "" "${luasrc}")
