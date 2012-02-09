SET(src)
SET(luasrc init.lua types.lua)
  
ADD_TORCH_PACKAGE(wrap "${src}" "${luasrc}")
ADD_TORCH_DOK(dok wrap "Fundamentals" "Lua/C automatic wrapper generation" 1.4)
