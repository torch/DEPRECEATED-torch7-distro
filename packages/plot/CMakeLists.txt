SET(src)
SET(luasrc init.lua gnuplot.lua )
  
ADD_TORCH_PACKAGE(plot "${src}" "${luasrc}" "Plot")
ADD_TORCH_DOK(dok plot "Fundamentals" "Plotting with Gnuplot" 1.)

##TARGET_LINK_LIBRARIES(plot luaT TH)
