setTargetNameToFolderName(${CMAKE_CURRENT_LIST_DIR})

set(linkDirectories ${CMAKE_SOURCE_DIR}/lib/CudaSampleCommon/lib/x64)
set(libsToLink freeglut.lib glew64.lib)
set(includeDirectories ${CMAKE_SOURCE_DIR}/lib/CudaSampleCommon/inc)

#glob files and save in MY_SRCS
include(CMakePackage.cmake)

enable_language(CUDA)
buildExe(${targetName} "${MY_SRCS}" "${linkDirectories}" "${libsToLink}" "${includeDirectories}")
# groupTarget(${targetName} ${CppExamples})

# Specify the linking to 3rdParty libs
include(3rdPartyLinking.cmake)

# Copy Dlls of non-global libraries
copy_dll(${targetName} ${CMAKE_SOURCE_DIR}/lib/CudaSampleCommon/bin/freeglut.dll)
copy_dll(${targetName} ${CMAKE_SOURCE_DIR}/lib/CudaSampleCommon/bin/glew64.dll)