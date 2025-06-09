cmake_minimum_required(VERSION 3.10)
project(thread_pool)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Boost REQUIRED COMPONENTS thread)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# Enable testing
include(CTest)
enable_testing()

# Thread pool library
add_library(thread_pool thread-pool.cpp)
target_include_directories(thread_pool 
    PUBLIC 
    ${CMAKE_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)

# Test executable
add_executable(thread_pool_test test/thread_pool_test.cpp)
target_link_libraries(thread_pool_test 
    PRIVATE 
    thread_pool
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    Boost::thread
)

# Register test
add_test(NAME thread_pool_tests COMMAND thread_pool_test)
