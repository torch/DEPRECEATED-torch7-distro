# Workaround: CMake sux if we do not create the directories
# This is completely incoherent compared to INSTALL(FILES ...)
FILE(MAKE_DIRECTORY "${Torch_BINARY_DIR}/dok")
INSTALL(DIRECTORY "${Torch_BINARY_DIR}/dok/" DESTINATION "${Torch_INSTALL_DOK_SUBDIR}")

ADD_CUSTOM_TARGET(documentation-dok
  ALL
  COMMENT "Built documentation")

MACRO(ADD_TORCH_DOK package section title rank)

  # Files for HTML creation
  SET(TORCH_DOK_HTML_TEMPLATE "${Torch_SOURCE_DIR}/scripts/doctemplate.html"
    CACHE FILEPATH "List of files needed for HTML doc creation")
  
  SET(TORCH_DOK_HTML_FILES "${Torch_SOURCE_DIR}/scripts/doctorch.css;${Torch_SOURCE_DIR}/scripts/torchlogo.png"
    CACHE STRING "HTML template needed for HTML doc creation")

  MARK_AS_ADVANCED(TORCH_DOK_HTML_FILES TORCH_DOK_HTML_TEMPLATE)

  SET(dokdstdir "${Torch_BINARY_DIR}/dok/${package}")
  SET(htmldstdir "${Torch_BINARY_DIR}/html/${package}")

  FILE(MAKE_DIRECTORY "${dokdstdir}")
  FILE(MAKE_DIRECTORY "${htmldstdir}")

  # Note: subdirectories are not handled (yet?)
  # http://www.cmake.org/pipermail/cmake/2008-February/020114.html
  FILE(GLOB dokfiles "help/*")

  SET(generatedfiles)
  FOREACH(dokfile ${dokfiles})
    GET_FILENAME_COMPONENT(_ext_ "${dokfile}" EXT)
    GET_FILENAME_COMPONENT(_file_ "${dokfile}" NAME_WE)

    IF(_ext_ STREQUAL ".dok")
      ADD_CUSTOM_COMMAND(OUTPUT "${dokdstdir}/${_file_}.dok" "${htmldstdir}/${_file_}.html"
        COMMAND  ${LUA_EXECUTABLE} ARGS "${Torch_SOURCE_DIR}/scripts/dokparse.lua" "${Torch_SOURCE_DIR}/packages/dok/init.lua" "${TORCH_DOK_HTML_TEMPLATE}" "${dokfile}" "${dokdstdir}/${_file_}.dok" "${htmldstdir}/${_file_}.html"
        DEPENDS ${LUA_EXECUTABLE}
        "${Torch_SOURCE_DIR}/packages/dok/init.lua"
        "${Torch_SOURCE_DIR}/scripts/dokparse.lua"
        "${dokfile}")
      
      SET(generatedfiles ${generatedfiles} "${dokdstdir}/${_file_}.dok" "${htmldstdir}/${_file_}.html")
    ELSE(_ext_ STREQUAL ".dok")
      ADD_CUSTOM_COMMAND(OUTPUT "${htmldstdir}/${_file_}${_ext_}"
        COMMAND ${CMAKE_COMMAND} ARGS "-E" "copy" "${dokfile}" "${htmldstdir}/${_file_}${_ext_}"
        DEPENDS "${dokfile}")
      SET(generatedfiles ${generatedfiles} "${htmldstdir}/${_file_}${_ext_}")
    ENDIF(_ext_ STREQUAL ".dok")
  ENDFOREACH(dokfile ${dokfiles})

  # copy extra files needed for HTML doc
  FOREACH(extrafile ${TORCH_DOK_HTML_FILES}) 
    GET_FILENAME_COMPONENT(_file_ "${extrafile}" NAME)
   
    ADD_CUSTOM_COMMAND(OUTPUT "${htmldstdir}/${_file_}"
      COMMAND ${CMAKE_COMMAND} ARGS "-E" "copy" "${extrafile}" "${htmldstdir}/${_file_}"
      DEPENDS "${extrafile}")
    SET(generatedfiles ${generatedfiles} "${htmldstdir}/${_file_}")
  ENDFOREACH(extrafile ${TORCH_DOK_HTML_FILES}) 

  # the doc depends on all these files to be generated
  ADD_CUSTOM_TARGET(${package}-dok-files
    DEPENDS ${generatedfiles})
  ADD_DEPENDENCIES(documentation-dok ${package}-dok-files)

  # Build the dok index if the package contains an index.dok file
  IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/help/index.dok")    
    ADD_CUSTOM_TARGET(${package}-dok-index
      ${LUA_EXECUTABLE} "${Torch_SOURCE_DIR}/scripts/dokindex.lua" "${TORCH_BINARY_DIR}/dok/index.dok" "${package}" "${section}" "${title}" "${rank}"
      DEPENDS ${LUA_EXECUTABLE}
      "${Torch_SOURCE_DIR}/scripts/dokindex.lua"
      "${CMAKE_CURRENT_SOURCE_DIR}/help/index.dok")
    
    ADD_DEPENDENCIES(documentation-dok ${package}-dok-index)

  ENDIF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/help/index.dok")
  
ENDMACRO(ADD_TORCH_DOK)
