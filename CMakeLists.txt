cmake_minimum_required(VERSION 3.31)

# Ninja generator for C++20 module support
if(NOT CMAKE_GENERATOR)
    set(CMAKE_GENERATOR "Ninja")
endif()

project(monte_carlo)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


add_executable(mc)
target_sources(mc PRIVATE
    main.cpp
PRIVATE
FILE_SET CXX_MODULES
FILES
    modules/MarketData.cpp
)

target_link_libraries(mc PRIVATE)