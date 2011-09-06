SET(src)
SET(luasrc init.lua gnuplot.lua )
  
ADD_TORCH_PACKAGE(plot "${src}" "${luasrc}" "Plot")

##TARGET_LINK_LIBRARIES(plot luaT TH)
