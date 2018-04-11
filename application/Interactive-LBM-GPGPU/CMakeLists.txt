setTargetNameToFolderName(${CMAKE_CURRENT_LIST_DIR})

find_package(OpenGL REQUIRED)

set(libsToLink ${OPENGL_LIBRARIES})

set(includeDirectories ${GLFW_INC_PATH}
                       ${CMAKE_SOURCE_DIR}/3rdParty/glm
                       ${CMAKE_SOURCE_DIR}/3rdParty/glew/include)

####################################################################
                       
enable_language(CUDA)

# on macOS 'enable_language' only looks for the compiler.
if (NOT CUDA_FOUND)
   message(STATUS "CUDA Toolkit not found! Trying to find package...")
   find_package(CUDA REQUIRED)
endif()
                       
#glob files and save in MY_SRCS
include(CMakePackage.cmake)

# manually add glew to compilation
# TODO: make static lib
set(MY_SRCS "${MY_SRCS}" "${CMAKE_SOURCE_DIR}/3rdParty/glew/src/glew.c" )

buildExe(${targetName} "${MY_SRCS}" "${linkDirectories}" "${libsToLink}" "${includeDirectories}")

# Specify the linking to 3rdParty libs
include(3rdPartyLinking.cmake)