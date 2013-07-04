# - Find MD5
# This module looks for md5. This module defines the 
# following values:
#  MD5_EXECUTABLE: the full path to the md5 tool.
#  MD5_FOUND: True if md5 has been found.

#=============================================================================
# Copyright 2001-2009 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

FIND_PROGRAM(MD5_EXECUTABLE
  NAMES md5 md5sum
)

# handle the QUIETLY and REQUIRED arguments and set MD5_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MD5 DEFAULT_MSG MD5_EXECUTABLE)

MARK_AS_ADVANCED( MD5_EXECUTABLE )

# MD5 option is deprecated.
# use MD5_EXECUTABLE instead.
SET (MD5 ${MD5_EXECUTABLE} )
