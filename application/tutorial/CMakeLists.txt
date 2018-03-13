setTargetNameToFolderName(${CMAKE_CURRENT_LIST_DIR})

find_package(OpenGL REQUIRED)
#set(linkDirectories ${CMAKE_SOURCE_DIR})
set(libsToLink ${OPENGL_LIBRARIES})
set(includeDirectories ${OPENGL_INCLUDE_DIR})

if (APPLE)
   #set(libsToLink GLEW)
   set(includeDirectories ${CUDA_CUT_INCLUDE_DIR})
else()
   set(linkDirectories )
   set(includeDirectories ${CUDA_CUT_INCLUDE_DIR})
endif()

set(glfw_path C:/lib/glfw-3.2.1)

set(includeDirectories ${glfw_path}/include ${CMAKE_SOURCE_DIR}/3rdParty/glm ${CMAKE_SOURCE_DIR}/3rdParty/glew/include)
#set(includeDirectories ${CMAKE_SOURCE_DIR}/3rdParty/glm)

#message(WARNING "CMAKE_SOURCE_DIR: "${CMAKE_SOURCE_DIR})

#glob files and save in MY_SRCS
include(CMakePackage.cmake)

enable_language(CUDA)

# on macOS 'enable_language' only looks for the compiler.
if (NOT CUDA_FOUND)
   message(STATUS "CUDA Toolkit not found! Trying to find package...")
   find_package(CUDA REQUIRED)
endif()

set(MY_SRCS "${MY_SRCS}" "${CMAKE_SOURCE_DIR}/3rdParty/glew/src/glew.c" )
message(STATUS "MY SOURCES: " "${MY_SRCS}")
buildExe(${targetName} "${MY_SRCS}" "${linkDirectories}" "${libsToLink}" "${includeDirectories}")
# groupTarget(${targetName} ${CppExamples})

# Specify the linking to 3rdParty libs
include(3rdPartyLinking.cmake)



# Copy Dlls of non-global libraries
#copy_dll(${targetName} ${CMAKE_SOURCE_DIR}/lib/CudaSampleCommon/bin/freeglut.dll)
#copy_dll(${targetName} ${CMAKE_SOURCE_DIR}/lib/CudaSampleCommon/bin/glew64.dll)