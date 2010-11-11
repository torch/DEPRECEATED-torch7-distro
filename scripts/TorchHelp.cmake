SET(Torch_HELP_INIT_LUA "${Torch_BINARY_DIR}/packages/help/init.lua")
SET(Torch_HLP2HTML_LUA "${Torch_BINARY_DIR}/scripts/hlp2html.lua")
CONFIGURE_FILE("scripts/hlp2html.lua.in" "${Torch_HLP2HTML_LUA}")

SET(HTML_DOC OFF CACHE BOOL "Create HTML documentation?")

# Workaround: CMake sux if we do not create the directories
# This is completely incoherent compared to INSTALL(FILES ...)
FILE(MAKE_DIRECTORY "${Torch_BINARY_DIR}/hlp")
INSTALL(DIRECTORY "${Torch_BINARY_DIR}/hlp/" DESTINATION "${Torch_INSTALL_HLP_SUBDIR}")

IF(HTML_DOC)
  FILE(MAKE_DIRECTORY "${Torch_BINARY_DIR}/html")
  INSTALL(DIRECTORY "${Torch_BINARY_DIR}/html/" DESTINATION "${Torch_INSTALL_HTML_SUBDIR}")
ENDIF(HTML_DOC)

ADD_CUSTOM_TARGET(hlp-help
  ALL
  COMMENT "Built .hlp help")

IF(HTML_DOC)
  ADD_CUSTOM_TARGET(html-help
    ALL
    COMMENT "Built .html help")
  ADD_DEPENDENCIES(html-help hlp-help)
ENDIF(HTML_DOC)

# internal helpful macro
MACRO(ADD_TORCH_HELP_FILE hlpfile hlpdstdir htmldstdir upindexhtml)

  IF(WEBSITE)
    SET(DOC_TEMPLATE "${Torch_SOURCE_DIR}/scripts/docwebtemplate.html")
  ELSE(WEBSITE)
    SET(DOC_TEMPLATE "${Torch_SOURCE_DIR}/scripts/doctemplate.html")
  ENDIF(WEBSITE)

  GET_FILENAME_COMPONENT(_file_ "${hlpfile}" NAME_WE)

  # copy .hlp files
  ADD_CUSTOM_COMMAND(OUTPUT "${hlpdstdir}/${_file_}.hlp"
    COMMAND ${CMAKE_COMMAND} ARGS "-E" "copy" "${hlpfile}" "${hlpdstdir}/${_file_}.hlp"
    DEPENDS "${hlpfile}")
  SET(generatedhlpfiles ${generatedhlpfiles} "${hlpdstdir}/${_file_}.hlp")

  IF(HTML_DOC)  
    # generate corresponding .html
    ADD_CUSTOM_COMMAND(OUTPUT "${htmldstdir}/${_file_}.html"
      COMMAND ${LUA_EXECUTABLE}
      ARGS "${Torch_HLP2HTML_LUA}" "${hlpdstdir}/${_file_}.hlp" "${DOC_TEMPLATE}" \"<a href=\\\"${upindexhtml}\\\">Torch Manual</a>\" "${htmldstdir}" "${_file__}.hlp" "${Torch_SOURCE_DIR}/scripts/docfilters.lua"
      DEPENDS "${LUA_EXECUTABLE}" "${hlpdstdir}/${_file_}.hlp" "${DOC_TEMPLATE}")
    SET(generatedhtmlfiles ${generatedhtmlfiles} "${htmldstdir}/${_file_}.html")
    
    # copy CSS file
    ADD_CUSTOM_COMMAND(OUTPUT "${htmldstdir}/doctorch.css"
      COMMAND ${CMAKE_COMMAND} ARGS "-E" "copy" "${Torch_SOURCE_DIR}/scripts/doctorch.css" "${htmldstdir}/doctorch.css"
      DEPENDS "${Torch_SOURCE_DIR}/scripts/doctorch.css")
    SET(generatedhlpfiles ${generatedhlpfiles} "${htmldstdir}/doctorch.css")
    
    # copy Torch logo file
    ADD_CUSTOM_COMMAND(OUTPUT "${htmldstdir}/torchlogo.png"
      COMMAND ${CMAKE_COMMAND} ARGS "-E" "copy" "${Torch_SOURCE_DIR}/scripts/torchlogo.png" "${htmldstdir}/torchlogo.png"
      DEPENDS "${Torch_SOURCE_DIR}/scripts/torchlogo.png")
    SET(generatedhlpfiles ${generatedhlpfiles} "${htmldstdir}/torchlogo.png")
  ENDIF(HTML_DOC)

ENDMACRO(ADD_TORCH_HELP_FILE)

MACRO(ADD_TORCH_HELP package)

  SET(generatedhlpfiles)
  FILE(MAKE_DIRECTORY "${Torch_BINARY_DIR}/hlp/${package}")

  IF(HTML_DOC)
    SET(generatedhtmlfiles)
    FILE(MAKE_DIRECTORY "${Torch_BINARY_DIR}/html/${package}")
  ENDIF(HTML_DOC)

  FILE(GLOB hlpfiles "help/*.hlp")
  FOREACH(hlpfile ${hlpfiles})
    ADD_TORCH_HELP_FILE("${hlpfiles}" "${Torch_BINARY_DIR}/hlp/${package}" "${Torch_BINARY_DIR}/html/${package}" "../index.html")
  ENDFOREACH(hlpfile ${hlpfiles})

  # Extra stuff to do if it is the help package
  IF(${package} STREQUAL "help")
    FILE(GLOB hlpfiles ${CMAKE_CURRENT_SOURCE_DIR}/main/*.hlp)
    FOREACH(hlpfile ${hlpfiles})
      IF(NOT ${hlpfile} STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/main/index.hlp")
        ADD_TORCH_HELP_FILE("${hlpfile}" "${Torch_BINARY_DIR}/hlp" "${Torch_BINARY_DIR}/html" "index.html")
      ENDIF(NOT ${hlpfile} STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/main/index.hlp")
    ENDFOREACH(hlpfile)
  ENDIF(${package} STREQUAL "help")

  ADD_CUSTOM_TARGET(${package}-hlp-files
    DEPENDS ${generatedhlpfiles})
  ADD_DEPENDENCIES(hlp-help ${package}-hlp-files)

  IF(HTML_DOC)
    ADD_CUSTOM_TARGET(${package}-html-files
      DEPENDS ${generatedhtmlfiles})
    ADD_DEPENDENCIES(html-help ${package}-html-files)
  ENDIF(HTML_DOC)


  # Build the help index if the package contains an index.hlp file
  IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/help/index.hlp")    

    ADD_CUSTOM_TARGET(${package}-hlp-index
      ${LUA_EXECUTABLE} "${Torch_SOURCE_DIR}/scripts/buildMainHelpIndex.lua" "${Torch_SOURCE_DIR}/packages/help/main/index.hlp" "${Torch_BINARY_DIR}/hlp/index.hlp" "${Torch_BINARY_DIR}/hlpsections.txt" "${package}" "${CMAKE_CURRENT_SOURCE_DIR}/help/index.hlp" "${package}/index.hlp" "${ARGN}"
      DEPENDS ${LUA_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/help/index.hlp" "${Torch_SOURCE_DIR}/packages/help/main/index.hlp")

    IF(HTML_DOC)
      ADD_CUSTOM_TARGET(${package}-html-index
        ${LUA_EXECUTABLE} "${Torch_HLP2HTML_LUA}" "${Torch_BINARY_DIR}/hlp/index.hlp" "${DOC_TEMPLATE}" \"\" "${Torch_BINARY_DIR}/html" "${Torch_BINARY_DIR}/hlp/index.hlp" "${Torch_SOURCE_DIR}/scripts/docfilters.lua"
        DEPENDS "${LUA_EXECUTABLE}" "${DOC_TEMPLATE}")
      
      ADD_DEPENDENCIES(${package}-html-index ${package}-hlp-index)
      ADD_DEPENDENCIES(${package}-html-files ${package}-html-index)
    ENDIF(HTML_DOC)
    ADD_DEPENDENCIES(${package}-hlp-files ${package}-hlp-index)

  ENDIF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/help/index.hlp")

ENDMACRO(ADD_TORCH_HELP)
