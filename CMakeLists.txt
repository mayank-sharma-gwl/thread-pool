cmake_minimum_required(VERSION 3.10)
project(thread_pool)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default package locations (customize these paths if needed)
# These can be overridden by providing -DBOOST_ROOT or -DGTEST_ROOT when invoking CMake.
if(NOT DEFINED BOOST_ROOT)
    set(BOOST_ROOT "/opt/homebrew/opt/boost")
endif()
if(NOT DEFINED GTEST_ROOT)
    set(GTEST_ROOT "/opt/homebrew/opt/googletest")
endif()

set(BOOST_INCLUDE_DIR ${BOOST_ROOT}/include)
set(BOOST_LIBRARY_THREAD ${BOOST_ROOT}/lib/libboost_thread.a)

set(GTEST_INCLUDE_DIR ${GTEST_ROOT}/include)
set(GTEST_LIBRARY ${GTEST_ROOT}/lib/libgtest.a)
set(GTEST_MAIN_LIBRARY ${GTEST_ROOT}/lib/libgtest_main.a)

# Check and install Boost if directory does not exist
if(NOT EXISTS ${BOOST_ROOT})
    message(STATUS "Boost not found at ${BOOST_ROOT}, attempting installation...")
    if(APPLE)
        execute_process(COMMAND brew install boost
                        RESULT_VARIABLE boost_install_result
                        OUTPUT_VARIABLE boost_install_output
                        ERROR_VARIABLE boost_install_error)
    elseif(UNIX)
        execute_process(COMMAND sh -c "sudo apt-get update && sudo apt-get install -y libboost-all-dev"
                        RESULT_VARIABLE boost_install_result
                        OUTPUT_VARIABLE boost_install_output
                        ERROR_VARIABLE boost_install_error)
    elseif(WIN32)
        message(FATAL_ERROR "Boost not found. Please install Boost manually on Windows.")
    endif()
    if(NOT boost_install_result EQUAL 0)
        message(FATAL_ERROR "Boost installation failed: ${boost_install_error}")
    endif()
endif()

# Check and install GoogleTest if directory does not exist
if(NOT EXISTS ${GTEST_ROOT})
    message(STATUS "GoogleTest not found at ${GTEST_ROOT}, attempting installation...")
    if(APPLE)
        execute_process(COMMAND brew install googletest
                        RESULT_VARIABLE gtest_install_result
                        OUTPUT_VARIABLE gtest_install_output
                        ERROR_VARIABLE gtest_install_error)
    elseif(UNIX)
        execute_process(COMMAND sh -c "sudo apt-get update && sudo apt-get install -y libgtest-dev"
                        RESULT_VARIABLE gtest_install_result
                        OUTPUT_VARIABLE gtest_install_output
                        ERROR_VARIABLE gtest_install_error)
    elseif(WIN32)
        message(FATAL_ERROR "GoogleTest not found. Please install GoogleTest manually on Windows.")
    endif()
    if(NOT gtest_install_result EQUAL 0)
        message(FATAL_ERROR "GoogleTest installation failed: ${gtest_install_error}")
    endif()
endif()

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
