cmake_minimum_required(VERSION 3.0)

project(fabric_defect_server)

set(CMAKE_BUILD_TYPE "Release")

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)

# check memory leak
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -fno-omit-frame-pointer")

add_definitions(-O2)

# pass img path or base64 data
# add_definitions(-D USE_PATH)

# pass json or binary data
add_definitions(-D PASS_BINARY)

# base64
find_package(base64)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/home/twc/TensorRT-8.0.1.6/include)
link_directories(/home/twc/TensorRT-8.0.1.6/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# opencv
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# websocketpp
find_package(Boost REQUIRED COMPONENTS system)
find_package(websocketpp)
include_directories(${WEBSOCKETPP_INCLUDE_DIR})

# libjpeg-turbo
include_directories(/opt/libjpeg-turbo/include)
link_directories(/opt/libjpeg-turbo/lib64)

#spdlog
find_package(spdlog)

#yaml-cpp
find_package(yaml-cpp)

add_executable(${PROJECT_NAME} src/detector.cpp src/manager.cpp src/main.cpp)
target_link_libraries(${PROJECT_NAME} pthread nvinfer cudart ${Boost_LIBRARIES} ${OpenCV_LIBS} turbojpeg base64 crypto spdlog::spdlog yaml-cpp)


