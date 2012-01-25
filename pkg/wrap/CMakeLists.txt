SET(src)
SET(luasrc init.lua types.lua)
  
ADD_TORCH_PACKAGE(wrap "${src}" "${luasrc}")
#ADD_TORCH_DOK(dok gnuplot "Fundamentals" "Plotting with Gnuplot" 1.)
