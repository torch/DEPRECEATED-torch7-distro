SET(src random.c)
SET(luasrc init.lua)

ADD_TORCH_PACKAGE(random "${src}" "${luasrc}" "Maths")

TARGET_LINK_LIBRARIES(random luaT TH)
