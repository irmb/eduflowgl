cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

if(POLICY CMP0042)
   CMAKE_POLICY(SET CMP0042 NEW)
endif()
if(POLICY CMP0020)
   CMAKE_POLICY(SET CMP0020 NEW)
endif()
if(POLICY CMP0028)
   CMAKE_POLICY(SET CMP0028 NEW)
endif()
if(POLICY CMP0037)
   CMAKE_POLICY(SET CMP0037 NEW)
endif()
if(POLICY CMP0047)
   CMAKE_POLICY(SET CMP0047 NEW)
endif()
if(POLICY CMP0053)
   CMAKE_POLICY(SET CMP0053 NEW)
endif()
if(POLICY CMP0054)
   CMAKE_POLICY(SET CMP0054 NEW)
endif()


if(APPLE)
   set(CMAKE_CXX_FLAGS -std=c++11)
   add_definitions("-Wno-c++11-extensions")
endif()


#############################################################
###                     ENVIRONMENT                       ###
#############################################################
set(cmakeMacroPath "CMakeMacros")
include(${cmakeMacroPath}/Environment/environment.cmake)

#############################################################
###                   GENERAL MACROS                      ###
#############################################################
include(${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/general/BuildTarget.cmake)
include(${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/general/BuildTargetUtilities.cmake)
include(${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/general/EndingsToCollect.cmake)
include(${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/general/FileUtilities.cmake)
include(${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/general/CopyDll.cmake)

#############################################################
###                   GENERAL SETTINGS                    ###
#############################################################
project(Interactive-LBM-GPGPU)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")
sharedLibs()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#############################################################
###                   SUBDIRECTORIES                      ###
#############################################################

add_subdirectory(application/InteractiveGlut)
