# -*- cmake -*-

INCLUDE_DIRECTORIES("${Torch_SOURCE_DIR}/libraries/TH")
INCLUDE_DIRECTORIES("${Torch_SOURCE_DIR}/libraries/luaT")
INCLUDE_DIRECTORIES(".")
INCLUDE(${QT_USE_FILE})		     

SET(qttorch_SRC qttorch.h qttorch.cpp)

MACRO_INSTALL_QTLUA_FILES(qttorch init.lua)

MACRO_ADD_QTLUA_MODULE(libqttorch ${qttorch_SRC})

TARGET_LINK_LIBRARIES(libqttorch luaT TH)

ADD_TORCH_DOK(dok qttorch "QT Interface" "Torch Tensor interface" 6.)
