include (${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/Cuda/Link.cmake)
linkCuda()

include (${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/OpenGL/Link.cmake)
linkOpenGL(${targetName})