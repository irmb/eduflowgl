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
    target_link_libraries(${targetName} "${COCOA_LIBRARY}" "${QuartzCore_LIBRARY}" "${IOKit_LIBRARY}")
    
    #CMake requires absolut pathes here for libraries that a not in LD_Library_PATH.
    #TODO: create marcros for:
    # GLFW (available in source code with CMake build files)
    target_link_libraries(${targetName} "${GLFW_BIN_PATH}/libglfw3.a")
else()
    target_link_libraries(${targetName} "${GLFW_BIN_PATH}/glfw3.lib")
endif()
