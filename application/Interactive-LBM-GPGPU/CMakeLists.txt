setTargetNameToFolderName(${CMAKE_CURRENT_LIST_DIR})

find_package(OpenGL REQUIRED)
# Find SDL2 package
# find_package(GLFW REQUIRED)
# find_package(SDL2 REQUIRED)

# Include SDL2 headers
# include_directories(${GLFW_INCLUDE_DIRS})








# set(SDL2_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/lib/x64/SDL2.lib;${CMAKE_CURRENT_LIST_DIR}/lib/x64/SDL2main.lib")

# link_directories(${CMAKE_SOURCE_DIR}/3rdparty/glfw-3.4.bin.WIN64/lib-vc2022)
# link_directories(${CMAKE_SOURCE_DIR}/erdparty/SDL2-2.30.0/lib)
if(WIN32)
set(includeDirectories ${GLFW_INCLUDE_DIRS}
                       ${CMAKE_SOURCE_DIR}/3rdparty/glm
                       ${CMAKE_SOURCE_DIR}/3rdparty/SDL2-2.30.0/include
                       ${CMAKE_SOURCE_DIR}/3rdparty/glfw-3.4.bin.WIN64/include
                       ${CMAKE_SOURCE_DIR}/3rdparty/glew/include
                       ${CMAKE_SOURCE_DIR}/3rdparty/CudaSampleCommon/inc) ###############
elseif(UNIX)
find_package(SDL2 REQUIRED)

# Include SDL2 headers
include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${SDL2_INCLUDE_DIRS})
set(libsToLink ${OPENGL_LIBRARIES}) 
set(includeDirectories ${GLFW_INCLUDE_DIRS}
                       ${CMAKE_SOURCE_DIR}/3rdparty/glm
                       ${CMAKE_SOURCE_DIR}/3rdparty/glew/include
                       ${CMAKE_SOURCE_DIR}/3rdparty/CudaSampleCommon/inc)
endif()

####################################################################
# set(libsToLink "/usr/lib/x86_64-linux-gnu/libSDL2.so")
# target_link_libraries(${targetName} "C:/Specialization Project/eduflowgl-GLFW/eduflowgl-GLFW/3rdparty/SDL2-2.30.0/lib/x64/SDL2.lib")
# ${CMAKE_SOURCE_DIR}/3rdparty/SDL2-2.30.0/include
# ${CMAKE_SOURCE_DIR}/3rdparty/glfw-3.4.bin.WIN64/include
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
set(MY_SRCS "${MY_SRCS}" "${CMAKE_SOURCE_DIR}/3rdparty/glew/src/glew.c" )

buildExe(${targetName} "${MY_SRCS}" "${linkDirectories}" "${libsToLink}" "${includeDirectories}")
copy_dll(${targetName} ${dllPath})
# Specify the linking to 3rdParty libs
include(3rdPartyLinking.cmake)

# C:/Users/izhma/OneDrive/Desktop/Untitled.bmp


