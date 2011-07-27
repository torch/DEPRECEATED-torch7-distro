SET(src init.c)
FILE(GLOB luasrc *.lua)

ADD_TORCH_PACKAGE(nn "${src}" "${luasrc}" "Machine Learning")

TARGET_LINK_LIBRARIES(nn luaT TH)
