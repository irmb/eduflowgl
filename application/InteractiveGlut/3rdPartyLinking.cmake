include (${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/Cuda/Link.cmake)
linkCuda()

include (${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/OpenGL/Link.cmake)
linkOpenGL(${targetName})

if(APPLE)
    find_library(GLUT_LIBRARY GLUT)
    #set(CMAKE_EXE_LINKER_FLAGS -libGLEW.a CACHE PATH "" FORCE)
    target_link_libraries(${targetName}  "GLEW" "${GLUT_LIBRARY}")
    #target_link_libraries(${targetName}  "${GLUT_LIBRARY}")
endif()