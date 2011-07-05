# -*- cmake -*-

INCLUDE(${QT_USE_FILE})		     
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")

SET(qtgui_SRC qtgui.h qtgui.cpp qtluagui.h qtluagui.cpp)

MACRO_INSTALL_QTLUA_FILES(qtgui init.lua)

MACRO_QT4_AUTOGEN(qtgui_GEN ${qtgui_SRC})
MACRO_ADD_QTLUA_MODULE(libqtgui ${qtgui_SRC} ${qtgui_GEN})

ADD_TORCH_HELP(qtgui "QT Interface")
