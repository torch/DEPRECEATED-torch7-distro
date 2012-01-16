
INCLUDE_DIRECTORIES("${Torch_SOURCE_DIR}/libraries/TH")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/THOmp")

# List of C/C++ sources
SET(src init.c openmp.c torchOmp.c nnOmp.c)

# List of Lua sources
SET(luasrc init.lua test/test.lua)

# Add a Torch package
ADD_TORCH_PACKAGE(openmp "${src}" "${luasrc}")

# You probably want this if you use luaT and TH libraries
TARGET_LINK_LIBRARIES(openmp luaT TH THOmp)
