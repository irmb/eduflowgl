setTargetNameToFolderName(${CMAKE_CURRENT_LIST_DIR})

find_package(OpenGL REQUIRED)
#set(linkDirectories ${CMAKE_SOURCE_DIR})
set(libsToLink ${OPENGL_LIBRARIES})
set(includeDirectories ${OPENGL_INCLUDE_DIR})

if (APPLE)
   set(linkDirectories ${CUDA_CUT_INCLUDE_DIR}/../lib/darwin/)
   #set(libsToLink GLEW)
   set(includeDirectories ${CUDA_CUT_INCLUDE_DIR})
else()
   set(linkDirectories ${CMAKE_SOURCE_DIR}/lib/CudaSampleCommon/lib/x64)
   set(libsToLink freeglut.lib glew64.lib)
   set(includeDirectories ${CMAKE_SOURCE_DIR}/lib/CudaSampleCommon/inc)
endif()

#message(WARNING "CMAKE_SOURCE_DIR: "${CMAKE_SOURCE_DIR})

#glob files and save in MY_SRCS
include(CMakePackage.cmake)

enable_language(CUDA)

# on macOS 'enable_language' only looks for the compiler.
if (NOT CUDA_FOUND)
   message(STATUS "CUDA Toolkit not found! Trying to find package...")
   find_package(CUDA REQUIRED)
endif()

buildExe(${targetName} "${MY_SRCS}" "${linkDirectories}" "${libsToLink}" "${includeDirectories}")
# groupTarget(${targetName} ${CppExamples})



# Specify the linking to 3rdParty libs
include(3rdPartyLinking.cmake)

# Copy Dlls of non-global libraries
copy_dll(${targetName} ${CMAKE_SOURCE_DIR}/lib/CudaSampleCommon/bin/freeglut.dll)
copy_dll(${targetName} ${CMAKE_SOURCE_DIR}/lib/CudaSampleCommon/bin/glew64.dll)