# Workaround: CMake sux if we do not create the directories
# This is completely incoherent compared to INSTALL(FILES ...)
MACRO(ADD_TORCH_HELP)
ENDMACRO(ADD_TORCH_HELP)

FILE(MAKE_DIRECTORY "${Torch_BINARY_DIR}/dok")
INSTALL(DIRECTORY "${Torch_BINARY_DIR}/dok/" DESTINATION "${Torch_INSTALL_DOK_SUBDIR}")

ADD_CUSTOM_TARGET(documentation-dok
  ALL
  COMMENT "Built documentation")

MACRO(ADD_TORCH_DOK package section title rank)

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
        COMMAND  ${LUA_EXECUTABLE} ARGS "${Torch_SOURCE_DIR}/scripts/dokparse.lua" "${Torch_SOURCE_DIR}/packages/dok/init.lua" "${Torch_SOURCE_DIR}/scripts/doctemplate.html" "${dokfile}" "${dokdstdir}/${_file_}.dok" "${htmldstdir}/${_file_}.html"
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

  # copy CSS file
  ADD_CUSTOM_COMMAND(OUTPUT "${htmldstdir}/doctorch.css"
    COMMAND ${CMAKE_COMMAND} ARGS "-E" "copy" "${Torch_SOURCE_DIR}/scripts/doctorch.css" "${htmldstdir}/doctorch.css"
    DEPENDS "${Torch_SOURCE_DIR}/scripts/doctorch.css")
  SET(generatedfiles ${generatedfiles} "${htmldstdir}/doctorch.css")
  
  # copy Torch logo file
  ADD_CUSTOM_COMMAND(OUTPUT "${htmldstdir}/torchlogo.png"
    COMMAND ${CMAKE_COMMAND} ARGS "-E" "copy" "${Torch_SOURCE_DIR}/scripts/torchlogo.png" "${htmldstdir}/torchlogo.png"
    DEPENDS "${Torch_SOURCE_DIR}/scripts/torchlogo.png")
  SET(generatedfiles ${generatedfiles} "${htmldstdir}/torchlogo.png")

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
