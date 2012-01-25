SET(src)
SET(luasrc init.lua)

ADD_TORCH_PACKAGE(lab "${src}" "${luasrc}" "Lab")
ADD_TORCH_DOK(dok lab "Fundamentals" "Matlab-like operations" 1.2)
