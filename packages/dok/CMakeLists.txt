SET(luasrc init.lua inline.lua)

ADD_TORCH_DOK(doktutorial tutorial "Introduction" "Tutorial" 0.2)
ADD_TORCH_DOK(dokinstall install "Introduction" "Installation Instructions" 0.1)
ADD_TORCH_DOK(doklua lua "Introduction" "Why Lua?" 0.3)

ADD_TORCH_PACKAGE(dok "${src}" "${luasrc}")
