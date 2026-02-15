cmake_minimum_required(VERSION 3.16)
project(clarabel_cpp LANGUAGES CXX)

option(CLARABEL_USE_CUDA "Build with CUDA support" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_library(clarabel_cpp
    src/core/types.cpp
    src/cones/cones.cpp
    src/residuals/residuals.cpp
    src/kktsystem/kkt.cpp
    src/kktsolvers/solver.cpp
    src/presolver/presolver.cpp
    src/solver/solver.cpp
)

target_include_directories(clarabel_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CLARABEL_USE_CUDA)
    enable_language(CUDA)
    target_sources(clarabel_cpp PRIVATE src/cones/cones_cuda.cu)
    target_compile_definitions(clarabel_cpp PUBLIC CLARABEL_USE_CUDA)
endif()

add_subdirectory(tests)
