cmake_minimum_required(VERSION 2.8)
project(SLIC_DBSCAN_CUDA)
  
if(MSVC_IDE)
  set(OpenCV_STATIC OFF)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
else(MSVC_IDE)
  set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
endif(MSVC_IDE)

set(CMAKE_CXX_FLAGS "-m64 -std=c++11 -fPIC ${CMAKE_CXX_FLAGS}")
if(CMAKE_BUILE_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "-g -funwind-tables" ${CMAKE_CXX_FLAGS})
else()
  set(CMAKE_CXX_FLAGS "${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
endif()

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
endif(APPLE)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

list(APPEND "-std=c++11 -ftree-vectorize")

set(LIB_OBJECTS
	src/lib/VCoreEngine.cpp
	src/lib/VSegEngine.cpp
	src/lib/VClusterEngine.cu
	src/lib/VSegEngineGPU.cu
)

set(DEMO_APP_OBJECTS
	src/example/demo.cpp
)

cuda_add_library(sdbscan ${LIB_OBJECTS} OPTIONS -gencode arch=${CUDA_ARCH},code=${CUDA_ARCH})

if(BUILD_DEMO)
add_executable(demo ${DEMO_APP_OBJECTS})
target_link_libraries(demo sdbscan ${CUDA_LIBRARY} ${OpenCV_LIBS})
endif()
