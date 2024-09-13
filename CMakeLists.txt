cmake_minimum_required(VERSION 3.23)
set(CMAKE_CUDA_ARCHITECTURES 75)

project(fabric_defect_server LANGUAGES CXX CUDA)

#set(CMAKE_BUILD_TYPE "Release")

# cuda
set(CMAKE_PREFIX_PATH /usr/local/cuda-11.7/bin)
set(CUDA_VERSION 11.7)
find_package(CUDA 11.7 REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# C++ 标准
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS FALSE)

# check memory leak
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -fno-omit-frame-pointer")

#add_definitions(-O3)

# pass img path or base64 data
# add_definitions(-D USE_PATH)

# pass json or binary data
add_definitions(-D PASS_BINARY)

# base64
find_package(base64)

# tensorrt
include_directories(/home/ding/TensorRT-8.5.3.1/include)
link_directories(/home/ding/TensorRT-8.5.3.1/lib)
set(CMAKE_PREFIX_PATH /home/ding/TensorRT-8.5.3.1)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
if (TensorRT_FOUND)
    message(STATUS "TensorRT found")
    include_directories(${TenssorRT_INCLUDE_DIRS})
    link_directories(${TenssorRT_LIBRARY_DIRS})
else ()
    message(FATAL_ERROR "TensorRT not found")
endif ()
#include_directories(/home/ding/TensorRT-8.5.3.1/include)
#link_directories(/home/ding/TensorRT-8.5.3.1/lib)

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

#build
#    yolo_utils
add_library(yolo_utils SHARED
        src/utils/postprocess.cpp
        src/utils/preprocess.cu
)
target_link_libraries(yolo_utils PUBLIC ${OpenCV_LIBS})
set_target_properties(yolo_utils PROPERTIES CUDA_ARCHITECTURES "61;70;75")

#    yolo_plugin
add_library(yolo_plugin SHARED
        src/plugins/yoloPlugins.cpp
        src/plugins/yoloForward_nc.cu
)
target_include_directories(yolo_plugin PUBLIC ${TensorRT_INCLUDE_DIRS})
target_link_libraries(yolo_plugin PUBLIC ${TensorRT_LIBRARIES})
set_target_properties(yolo_plugin PROPERTIES CUDA_ARCHITECTURES "61;70;75")

#     build
add_executable(build
        src/build.cu
        ${TensorRT_SAMPLE_DIR}/common/logger.cpp
        ${TensorRT_SAMPLE_DIR}/common/sampleUtils.cpp
)
target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common)
target_link_libraries(build PRIVATE -Wl,--no-as-needed yolo_plugin) # -Wl,--no-as-needed is needed to avoid linking errors
target_link_libraries(build PRIVATE yolo_utils)
set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75")


# runtime
add_executable(${PROJECT_NAME} src/detector.cpp src/manager.cpp src/main.cpp)
target_link_libraries(${PROJECT_NAME}
        pthread
        nvinfer
        cudart
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        turbojpeg
        base64
        crypto
        spdlog::spdlog
        yaml-cpp)


