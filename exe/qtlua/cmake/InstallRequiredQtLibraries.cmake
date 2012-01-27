# -*- cmake -*-

# select debug or release versions
STRING(TOUPPER _debug "${CMAKE_BUILD_TYPE}")
IF (NOT "${_debug}" STREQUAL "DEBUG")
  SET(_debug "RELEASE")
ENDIF (NOT "${_debug}" STREQUAL "DEBUG")


# [from UseQt4.cmake] fix use variables
IF(QT_DONT_USE_QTGUI)
  SET(QT_USE_QTGUI 0)
ELSE(QT_DONT_USE_QTGUI)
  SET(QT_USE_QTGUI 1)
ENDIF(QT_DONT_USE_QTGUI)

IF(QT_DONT_USE_QTCORE)
  SET(QT_USE_QTCORE 0)
ELSE(QT_DONT_USE_QTCORE)
  SET(QT_USE_QTCORE 1)
ENDIF(QT_DONT_USE_QTCORE)

# [from UseQt4.cmake] list dependent modules, so their modules are automatically on
SET(QT_QT3SUPPORT_MODULE_DEPENDS QTGUI QTSQL QTXML QTNETWORK QTCORE)
SET(QT_QTSVG_MODULE_DEPENDS QTGUI QTXML QTCORE)
SET(QT_QTUITOOLS_MODULE_DEPENDS QTGUI QTXML QTCORE)
SET(QT_QTHELP_MODULE_DEPENDS QTGUI QTSQL QTXML QTCORE)
SET(QT_PHONON_MODULE_DEPENDS QTGUI QTDBUS QTCORE)
SET(QT_QTDBUS_MODULE_DEPENDS QTXML QTCORE)
SET(QT_QTXMLPATTERNS_MODULE_DEPENDS QTNETWORK QTCORE)
SET(QT_QTWEBKIT_MODULE_DEPENDS QTNETWORK PHONON QTCORE)

# [from UseQt4.cmake] Qt modules  (in order of dependence)
FOREACH(module QT3SUPPORT QTOPENGL QTASSISTANT QTDESIGNER QTMOTIF QTNSPLUGIN
               QTSCRIPT QTSVG QTUITOOLS QTHELP QTWEBKIT PHONON QTGUI QTTEST 
               QTDBUS QTXML QTSQL QTXMLPATTERNS QTNETWORK QTCORE)

  IF (QT_USE_${module})
    IF (QT_${module}_FOUND)
      FOREACH(_lib ${QT_${module}_LIBRARY_${_debug}} ${QT_${module}_LIB_DEPENDENCIES})
        GET_FILENAME_COMPONENT(_libname "${_lib}" NAME_WE)
        IF (EXISTS "${QT_BINARY_DIR}/${_libname}.dll")
          # --- DEBUG MESSAGE
          # MESSAGE("Will install ${QtLua_INSTALL_BIN_SUBDIR}/${_libname}.dll")
          INSTALL(PROGRAMS "${QT_BINARY_DIR}/${_libname}.dll"
            DESTINATION "${QtLua_INSTALL_BIN_SUBDIR}")
        ENDIF (EXISTS "${QT_BINARY_DIR}/${_libname}.dll")
      ENDFOREACH(_lib)
      FOREACH(depend_module ${QT_${module}_MODULE_DEPENDS})
        SET(QT_USE_${depend_module} 1)
      ENDFOREACH(depend_module ${QT_${module}_MODULE_DEPENDS})
    ENDIF (QT_${module}_FOUND)
  ENDIF (QT_USE_${module})
ENDFOREACH(module)

# Create qt.conf in torch binary directory
FILE(WRITE "${CMAKE_CURRENT_BINARY_DIR}/qt.conf" "[Paths]")
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
  DESTINATION "${QtLua_INSTALL_BIN_SUBDIR}")

# Install qt plugins
FOREACH(_pdir "codecs" "imageformats")
  IF (EXISTS "${QT_PLUGINS_DIR}/${_pdir}")
    FILE(GLOB _pdll  "${QT_PLUGINS_DIR}/${_pdir}/*.dll")
    FOREACH(_dll ${_pdll})
      GET_FILENAME_COMPONENT(_dll "${_dll}" NAME_WE)
      STRING(REGEX REPLACE "d4$" "4" _dll_nd "${_dll}")
      IF (NOT EXISTS "${QT_PLUGINS_DIR}/${_pdir}/${_dll_nd}.dll")
        SET(_dll_nd "${_dll}")
      ENDIF (NOT EXISTS "${QT_PLUGINS_DIR}/${_pdir}/${_dll_nd}.dll")
      STRING(REGEX REPLACE "4$" "d4" _dll_d "${_dll}")
      IF (NOT EXISTS "${QT_PLUGINS_DIR}/${_pdir}/${_dll_d}.dll")
        SET(_dll_d "${_dll}")
      ENDIF (NOT EXISTS "${QT_PLUGINS_DIR}/${_pdir}/${_dll_d}.dll")
      SET(_inst 0)
      IF ("${_debug}" STREQUAL "DEBUG" AND "${_dll}" STREQUAL "${_dll_d}")
        SET(_inst 1)
      ELSEIF (NOT "${_debug}" STREQUAL "DEBUG" AND "${_dll}" STREQUAL "${_dll_nd}")
        SET(_inst 1)
      ENDIF ("${_debug}" STREQUAL "DEBUG" AND "${_dll}" STREQUAL "${_dll_d}")
      IF (_inst)
        # --- DEBUG MESSAGE
        # MESSAGE("Will install ${QtLua_INSTALL_BIN_SUBDIR}/plugins/${_pdir}/${_dll}.dll")
        INSTALL(PROGRAMS "${QT_PLUGINS_DIR}/${_pdir}/${_dll}.dll"
          DESTINATION "${QtLua_INSTALL_BIN_SUBDIR}/plugins/${_pdir}" )
      ENDIF(_inst)
    ENDFOREACH(_dll)
  ENDIF (EXISTS "${QT_PLUGINS_DIR}/${_pdir}")
ENDFOREACH(_pdir)

