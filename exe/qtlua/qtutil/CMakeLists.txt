# -*- cmake -*-

SET(QTLUA_DEFINITIONS)
SET(QTLUA_INCLUDE_DIR 
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}")

ADD_DEFINITIONS( ${QTUTIL_DEFINITIONS} ${LUA_DEFINITIONS})
INCLUDE_DIRECTORIES(${QTUTIL_INCLUDE_DIR} ${LUA_INCLUDE_DIR})
INCLUDE(${QT_USE_FILE})		     


# --- compile library

SET(qtutil_SRC 
  "qmenuaction.h" "qmenuaction.cpp"
  "qthumbwheel.h" "qthumbwheel.cpp" )

MACRO_QT4_AUTOGEN(qtutil_GEN ${qtutil_SRC})

ADD_LIBRARY(libqtutil SHARED ${qtutil_SRC} ${qtutil_GEN})
TARGET_LINK_LIBRARIES(libqtutil ${LUA_LIBRARIES} ${QT_LIBRARIES})
SET_TARGET_PROPERTIES(libqtutil PROPERTIES 
  LINKER_LANGUAGE CXX 
  OUTPUT_NAME "qtutil" )

# --- install library and include files

INSTALL(TARGETS libqtutil
  RUNTIME DESTINATION ${Torch_INSTALL_BIN_SUBDIR} 
  LIBRARY DESTINATION ${Torch_INSTALL_LIB_SUBDIR} 
  ARCHIVE DESTINATION ${Torch_INSTALL_LIB_SUBDIR} )

# --- config for internal use

SET(QTUTIL_LIBRARIES "libqtutil")
SET(QTUTIL_DEFINITIONS)
CONFIGURE_FILE(QtUtilConfig.cmake.in 
  "${Torch_BINARY_DIR}/cmake/QtUtilConfig.cmake")
SET(QtUtil_DIR "${Torch_BINARY_DIR}/cmake" CACHE PATH 
  "Directory containing QtUtilConfig.cmake")
MARK_AS_ADVANCED(QtUtil_DIR)

# --- config for external use

GET_TARGET_PROPERTY(QTUTIL_OUTPUT_NAME libqtutil LOCATION)
GET_FILENAME_COMPONENT(QTUTIL_OUTPUT_NAME ${QTUTIL_OUTPUT_NAME} NAME)
SET(QTUTIL_LIBRARIES "${Torch_INSTALL_LIB}/${QTUTIL_OUTPUT_NAME}")
SET(QTUTIL_INCLUDE_DIR "${Torch_INSTALL_INCLUDE}/qtutil")
CONFIGURE_FILE("QtUtilConfig.cmake.in" 
  "${Torch_BINARY_DIR}/cmake-external/QtUtilConfig.cmake")
INSTALL(FILES "${Torch_BINARY_DIR}/cmake-external/QtUtilConfig.cmake" 
  DESTINATION "${Torch_INSTALL_CMAKE_SUBDIR}")

