cmake_minimum_required(VERSION 3.16)
project(DHS CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type defaults to Release if not specified.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0")

include_directories(${CMAKE_SOURCE_DIR}/include)

# --- Checkpoint 1 ---
add_executable(checkpoint1 src/checkpoint1.cpp)
target_include_directories(checkpoint1 PRIVATE ${CMAKE_SOURCE_DIR}/src)

# --- Checkpoint 2 ---
add_executable(checkpoint2 src/checkpoint2.cpp)
target_include_directories(checkpoint2 PRIVATE ${CMAKE_SOURCE_DIR}/src)

# --- Checkpoint 3 ---
add_executable(checkpoint3 src/checkpoint3.cpp)
target_include_directories(checkpoint3 PRIVATE ${CMAKE_SOURCE_DIR}/src)

# --- Checkpoint 4 ---
add_executable(checkpoint4 src/checkpoint4.cpp)
target_include_directories(checkpoint4 PRIVATE ${CMAKE_SOURCE_DIR}/src)
