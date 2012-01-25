# -*- cmake -*-

INCLUDE(${QT_USE_FILE})		     
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")

SET(qtcore_SRC qtcore.h qtcore.cpp)

MACRO_INSTALL_QTLUA_FILES(qtcore init.lua)

MACRO_ADD_QTLUA_MODULE(libqtcore ${qtcore_SRC})

ADD_TORCH_DOK(dok qtcore "QT Interface" "Core interface" 6.2)
