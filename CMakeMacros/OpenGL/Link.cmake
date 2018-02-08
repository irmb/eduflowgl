macro(linkOpenGL targetName)
    find_package(OpenGL REQUIRED)
	
	include_directories(${OPENGL_INCLUDE_DIRS})
	
	target_link_libraries(${targetName} ${OPENGL_LIBRARIES})
endmacro(linkOpenGL)
