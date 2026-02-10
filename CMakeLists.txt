cmake_minimum_required(VERSION 3.31)

# Ninja generator for C++20 module support
if(NOT CMAKE_GENERATOR)
    set(CMAKE_GENERATOR "Ninja")
endif()

project(monte_carlo)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(mc)
target_sources(mc PRIVATE
    main.cpp
PRIVATE
FILE_SET CXX_MODULES
FILES
    modules/MarketData.cpp
    modules/EquitySim.cpp
)

target_link_libraries(mc PRIVATE CURL::libcurl nlohmann_json::nlohmann_json)

# Enable SIMD optimizations (AVX2 + FMA)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mc PRIVATE -mavx2 -mfma -O3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(mc PRIVATE /arch:AVX2 /O2)
endif()