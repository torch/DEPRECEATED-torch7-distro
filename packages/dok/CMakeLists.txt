SET(luasrc init.lua inline.lua)

ADD_TORCH_PACKAGE(dok "${src}" "${luasrc}")
