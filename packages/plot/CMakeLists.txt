SET(src)
SET(luasrc init.lua gnuplot.lua )
  
ADD_TORCH_PACKAGE(plot "${src}" "${luasrc}" "Plot")
ADD_TORCH_DOK(dok plot "Fundamentals" "Plot" 1.)

##TARGET_LINK_LIBRARIES(plot luaT TH)
