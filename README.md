# cmakeTemplate
This repository is a common template for c++ projects with cmake.

## folder structure

**applications**
- every folder generate a lib or an executable
- the main is placed here
- every folder contains a CMakeList.txt with applications details like links and includes

**CMakeMacros**
- cmake macros to handle the project
- git subtree: original [Repository on Github](https://github.com/ClubOfDigitalEngineering/CMakeMacros)

**src**
- contains source code

**examples**
- AllTest
- Boost
- Cuda
- FFTW
- GMock
- JsonCpp
- Lib
- LibTest
- Metis
- MPI
- OpenMP
- Qt
- VTK
- Git + additional build Infos



## platforms
- this project is used by:
  - visual studio 2013
  - visual studio 2015
  - visual studio 2017
  - unix makefile
  - Xcode
  - clion
