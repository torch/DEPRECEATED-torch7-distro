# -*- cmake -*-
#
# MACRO_QT4_AUTOGEN(<var> <..infiles..>)
# examines the files <...infiles...> and produces rules
# to generate C++ files that should be compiled.
# The names of the generated files is added into variable <var>.
# The following cases are recognized (emulating qmake)
#
# 1- For each header files <name.h> containing the string Q_OBJECT 
#    a file named <moc_name.cxx> is generated using moc.
# 2- For each resource file <name.qrc>,
#    a file named <qrc_name.cxx> is generated using rcc.
# 3- For each source file <name.cpp> or <name.cxx> 
#    containing the string Q_OBJECT, a file named <name.moc>
#    is generated using <moc>. It is expected that the C++ file
#    includes this file.
# 4- For each designed file <name.ui>,
#    a file named <ui_name.h> is generated using uic.
#
# Copyright (c) 2007, Leon Bottou
#
# Redistribution and use is allowed according to the terms of the BSD license.

INCLUDE(MacroAddFileDependencies)

MACRO(MACRO_QT4_AUTOGEN outfiles)
  FOREACH (it ${ARGN})
    GET_FILENAME_COMPONENT(ext "${it}" EXT)
    IF ("${ext}" MATCHES "^\\.(ui)$")
        QT4_WRAP_UI(${outfiles} "${it}")
    ELSEIF ("${ext}" MATCHES "^\\.(qrc)$")
        QT4_ADD_RESOURCES(${outfiles} "${it}")
    ELSEIF ("${ext}" MATCHES "^\\.(h)$")
        GET_FILENAME_COMPONENT(abs "${it}" ABSOLUTE)
        IF (EXISTS "${abs}")
          FILE(READ "${abs}" _contents)
          STRING(REGEX MATCH "Q_OBJECT" _match "${_contents}")
          IF (_match)
            QT4_WRAP_CPP(${outfiles} "${it}")
          ENDIF (_match)
        ENDIF(EXISTS "${abs}")
    ELSEIF ("${ext}" MATCHES "^\\.(cpp|cxx)$") 
        GET_FILENAME_COMPONENT(abs "${it}" ABSOLUTE)
        GET_FILENAME_COMPONENT(nam "${it}" NAME_WE)
        IF (EXISTS "${abs}")
          FILE(READ "${abs}" _contents)
          STRING(REGEX MATCH "Q_OBJECT" _match "${_contents}")
          IF (_match)
            QT4_GENERATE_MOC("${abs}" "${CMAKE_CURRENT_BINARY_DIR}/${nam}.moc")
	    MACRO_ADD_FILE_DEPENDENCIES("${it}" "${CMAKE_CURRENT_BINARY_DIR}/${nam}.moc")
          ENDIF (_match)
        ENDIF(EXISTS "${abs}")
    ENDIF ("${ext}" MATCHES "^\\.(ui)$") 
  ENDFOREACH(it)
ENDMACRO(MACRO_QT4_AUTOGEN outfiles)