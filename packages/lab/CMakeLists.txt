SET(src lab.c utils.c init.c)
SET(luasrc init.lua hist.lua)
SET(luasrc ${luasrc} test/test.lua)
  
ADD_TORCH_PACKAGE(lab "${src}" "${luasrc}" "Lab")

TARGET_LINK_LIBRARIES(lab luaT TH)
