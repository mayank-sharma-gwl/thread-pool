cmake_minimum_required(VERSION 3.10)
project(thread_pool)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Mac ARM64-specific settings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -arch arm64")
set(CMAKE_OSX_ARCHITECTURES "arm64")

set(BOOST_ROOT /opt/homebrew/opt/boost)
set(BOOST_INCLUDE_DIR ${BOOST_ROOT}/include)
set(BOOST_LIBRARY_THREAD ${BOOST_ROOT}/lib/libboost_thread.a)

# Force set GTest paths (assuming Homebrew installation)
set(GTEST_ROOT /opt/homebrew/opt/googletest)
set(GTEST_INCLUDE_DIR ${GTEST_ROOT}/include)
set(GTEST_LIBRARY ${GTEST_ROOT}/lib/libgtest.a)
set(GTEST_MAIN_LIBRARY ${GTEST_ROOT}/lib/libgtest_main.a)

# Compiler settings
set(CMAKE_C_COMPILER /usr/bin/clang)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)

# Enable testing
include(CTest)
enable_testing()

# Find GTest package with forced paths
find_package(GTest REQUIRED PATHS ${GTEST_ROOT} NO_DEFAULT_PATH)
find_package(Threads REQUIRED)

# Thread pool library
add_library(thread_pool thread-pool.cpp)
target_include_directories(thread_pool 
    PUBLIC 
    ${CMAKE_SOURCE_DIR}
    ${GTEST_INCLUDE_DIR}
    ${BOOST_INCLUDE_DIR}
)

# Test executable
add_executable(thread_pool_test test/thread_pool_test.cpp)
target_link_libraries(thread_pool_test 
    PRIVATE 
    thread_pool
    ${GTEST_LIBRARY}
    ${GTEST_MAIN_LIBRARY}
    Threads::Threads
    ${BOOST_LIBRARY_THREAD}
)

# Register test
add_test(NAME thread_pool_tests COMMAND thread_pool_test)
