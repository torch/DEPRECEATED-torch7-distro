SET(src random.c)
SET(luasrc init.lua)

ADD_TORCH_PACKAGE(random "${src}" "${luasrc}" "Maths")
ADD_TORCH_DOK(dok random "Fundamentals" "Random numbers" 1.)

TARGET_LINK_LIBRARIES(random luaT TH)
