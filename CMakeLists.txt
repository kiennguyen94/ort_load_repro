cmake_minimum_required(VERSION 3.17)
project(test_load CXX)

set (CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCXXCompilerFlag)

add_executable(main ./main.cpp)

message(STATUS "ort install previx ${ORT_INSTALL_PREFIX}")
target_include_directories(main PUBLIC ${ORT_INSTALL_PREFIX}/include
    PUBLIC ${EIGEN_HDR_DIR})
target_link_libraries(main PUBLIC ${ORT_INSTALL_PREFIX}/lib64/libonnxruntime.so)
