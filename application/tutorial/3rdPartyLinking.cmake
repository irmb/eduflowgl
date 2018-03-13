include (${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/Cuda/Link.cmake)
linkCuda()

include (${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/OpenGL/Link.cmake)
linkOpenGL(${targetName})

if(APPLE)
    #link_directories(${CMAKE_SOURCE_DIR}/3rdparty/glfw/lib)
    #link_directories(${CMAKE_SOURCE_DIR}/3rdparty/glfw/lib ${CUDA_CUT_INCLUDE_DIR}/../lib/darwin/)
    #find_library(GLUT_LIBRARY GLUT)

    find_library(COCOA_LIBRARY COCOA)
    find_library(QuartzCore_LIBRARY QuartzCore)
    find_library(IOKit_LIBRARY IOKit)
    
    #CMake requires absolut pathes here for libraries that a not in LD_Library_PATH.
    #TODO: create marcros for:
    # GLFW (available in source code with CMake build files)
    # GLEW is included here
    target_link_libraries(${targetName} "${CMAKE_SOURCE_DIR}/3rdparty/glfw/lib/darwin/libglfw3.a" "${COCOA_LIBRARY}" "${QuartzCore_LIBRARY}" "${IOKit_LIBRARY}")
else()
    target_link_libraries(${targetName} "${glfw_path}/build/src/Release/glfw3.lib")
endif()
