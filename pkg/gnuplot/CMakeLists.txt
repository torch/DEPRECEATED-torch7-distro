SET(src)
SET(luasrc init.lua gnuplot.lua )
  
ADD_TORCH_PACKAGE(gnuplot "${src}" "${luasrc}")
ADD_TORCH_DOK(dok gnuplot "Fundamentals" "Plotting with Gnuplot" 1.)

##TARGET_LINK_LIBRARIES(plot luaT TH)
