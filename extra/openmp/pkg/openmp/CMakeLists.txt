SET(src init.c openmp.c torchOmp.c nnOmp.c)
SET(luasrc init.lua test/test.lua)

ADD_TORCH_PACKAGE(openmp "${src}" "${luasrc}")

TARGET_LINK_LIBRARIES(openmp luaT TH THOmp)
