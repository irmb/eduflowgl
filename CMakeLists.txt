cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
# cmake_policy(SET CMP0146 OLD)
cmake_policy(SET CMP0074 NEW)

find_package(CUDAToolkit REQUIRED)

set(CUDA_CUT_INCLUDE_DIR "/include")
# C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations
# set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3" CACHE PATH "Path to CUDA Toolkit")

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

cmake_policy(SET CMP0072 NEW)



if(APPLE)
   set(CMAKE_CXX_FLAGS -std=c++11 CACHE PATH "" FORCE)
   set(CMAKE_CUDA_FLAGS -std=c++11 CACHE PATH "" FORCE)
   set(CMAKE_VERBOSE_MAKEFILE ON CACHE PATH "" FORCE)

   #add_definitions("-Wno-c++11-extensions")
endif()


#############################################################
###                     ENVIRONMENT                       ###
#############################################################
set(cmakeMacroPath "CMakeMacros")
# include(${cmakeMacroPath}/Environment/environment.cmake)

#############################################################
###                   GENERAL MACROS                      ###
#############################################################
include(${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/general/BuildTarget.cmake)
include(${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/general/BuildTargetUtilities.cmake)
include(${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/general/EndingsToCollect.cmake)
include(${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/general/FileUtilities.cmake)
include(${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/general/CopyDll.cmake)
# include(${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/SDL/sdl2-config.cmake)
# include(3rdparty/SDL2-2.30.0/cmake/sdl2-config.cmake)
# include(${CMAKE_SOURCE_DIR}/${cmakeMacroPath}/SDL/sdl2-config-version.cmake)

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

add_subdirectory(application/Interactive-LBM-GPGPU)
