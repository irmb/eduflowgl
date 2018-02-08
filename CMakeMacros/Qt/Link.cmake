macro(linkQt targetName)

	find_package(Qt5Core REQUIRED)
	find_package(Qt5Widgets REQUIRED)
	find_package(Qt5Gui REQUIRED)
	find_package(Qt5PrintSupport REQUIRED)
	find_package(Qt5Charts REQUIRED)
    find_package(Qt5OpenGL REQUIRED)

	include_directories(${QT5Widgets_INCLUDES})
	include_directories(${QT5Core_INCLUDES})
	include_directories(${QT5Gui_INCLUDES})
	include_directories(${QT5PrintSupport_INCLUDES})
    include_directories(${QT5OpenGL_INCLUDES})
    
	add_definitions(${Qt5Widgets_DEFINITIONS})
	add_definitions(${Qt5Core_DEFINITIONS})
	add_definitions(${Qt5Gui_DEFINITIONS})
	add_definitions(${Qt5PrintSupport_DEFINITIONS})
    add_definitions(${QT5OpenGL_DEFINITIONS})

	target_link_libraries(${targetName} Qt5::Widgets)
	target_link_libraries(${targetName} Qt5::Core)
	target_link_libraries(${targetName} Qt5::PrintSupport)
	target_link_libraries(${targetName} Qt5::Charts)
	target_link_libraries(${targetName} Qt5::OpenGL)

endmacro(linkQt)
