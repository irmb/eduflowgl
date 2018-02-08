macro(copy_dll targetName dllPath)
    add_custom_command(TARGET ${targetName} POST_BUILD  # Adds a post-build event to MyTest
        COMMAND ${CMAKE_COMMAND} -E copy_if_different   # which executes "cmake - E copy_if_different..."
        "${dllPath}"                                   # <--this is in-file
        $<TARGET_FILE_DIR:${targetName}>)               # <--this is out-file path
endmacro(copy_dll)