cmake_minimum_required(VERSION 3.28)

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

# Optimization flags
option(ENABLE_NATIVE_TUNING "Enable -march=native optimization" OFF)

if(ENABLE_NATIVE_TUNING)
    if(MSVC)
        target_compile_options(mc PRIVATE /arch:AVX2 /O2)
    else()
        target_compile_options(mc PRIVATE -march=native -O3)
    endif()
else()
    # Default safe optimizations
    if(MSVC)
        target_compile_options(mc PRIVATE /O2)
    else()
        target_compile_options(mc PRIVATE -O3)
        
        # Check if AVX2 is available even if not native
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
        
        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(mc PRIVATE -mavx2)
        endif()
        if(COMPILER_SUPPORTS_FMA)
            target_compile_options(mc PRIVATE -mfma)
        endif()
    endif()
endif()