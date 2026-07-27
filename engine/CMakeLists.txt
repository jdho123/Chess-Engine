cmake_minimum_required(VERSION 3.20)
project(engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(engine
    src/main.cpp
    src/nnue.cpp
    src/nnue_kernels.cpp
    src/nnue_weights.cpp
    src/search.cpp
    src/clock.cpp
    src/engine.cpp
)

target_include_directories(engine PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(engine PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/chess/include
)

target_compile_options(engine PRIVATE -O3 -march=native)
