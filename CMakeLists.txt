set(CMAKE_C_COMPILER "/opt/homebrew/bin/gcc-14")
set(CMAKE_CXX_COMPILER "/opt/homebrew/bin/g++-14")
cmake_minimum_required(VERSION 3.10)
project(thread_pool)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native -flto -funroll-loops -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG -march=native -flto -funroll-loops -Wall -Wextra")

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")

# Set default package locations (customize these paths as needed)
# These can be overridden by providing -DBOOST_ROOT or -DGTEST_ROOT when invoking CMake.
if(NOT DEFINED BOOST_ROOT)
    if(APPLE)
        set(BOOST_ROOT "/opt/homebrew/opt/boost")
    elseif(UNIX)
        # On Ubuntu (GitHub CI), rely on system-installed packages.
        set(BOOST_ROOT "")
    endif()
endif()

if(NOT DEFINED GTEST_ROOT)
    if(APPLE)
        set(GTEST_ROOT "/opt/homebrew/opt/googletest")
    elseif(UNIX)
        # On Ubuntu, GTest is installed via apt, so leave unset.
        set(GTEST_ROOT "")
    endif()
endif()

if(BOOST_ROOT STREQUAL "")
    # Use system defaults found by find_package
    set(BOOST_INCLUDE_DIR "")
    set(BOOST_LIBRARY_THREAD "")
    # On Ubuntu/GitHub CI, use the system's Boost installation.
    find_package(Boost REQUIRED COMPONENTS thread)
    set(BOOST_INCLUDE_DIR ${Boost_INCLUDE_DIRS})
else()
    set(BOOST_INCLUDE_DIR ${BOOST_ROOT}/include)
    set(BOOST_LIBRARY_THREAD ${BOOST_ROOT}/lib/libboost_thread.a)
endif()

if(GTEST_ROOT STREQUAL "")
    # On Ubuntu, we'll use system paths.
    set(GTEST_INCLUDE_DIR "")
    set(GTEST_LIBRARY "")
    set(GTEST_MAIN_LIBRARY "")
else()
    set(GTEST_INCLUDE_DIR ${GTEST_ROOT}/include)
    set(GTEST_LIBRARY ${GTEST_ROOT}/lib/libgtest.a)
    set(GTEST_MAIN_LIBRARY ${GTEST_ROOT}/lib/libgtest_main.a)
endif()

if(APPLE OR WIN32)
    # For legacy platforms, check and install Boost if not found.
    if(NOT EXISTS ${BOOST_ROOT})
        message(STATUS "Boost not found at ${BOOST_ROOT}, attempting installation...")
        if(APPLE)
            execute_process(COMMAND brew install boost
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
endif()

if(APPLE)
    set(GTest_DIR "${GTEST_ROOT}/lib/cmake/GTest")
elseif(UNIX)
    # On Ubuntu, GTest is installed via apt and CMake package is located here.
    set(GTest_DIR "/usr/lib/cmake/gtest")
endif()

find_package(GTest REQUIRED PATHS ${GTEST_ROOT} ${GTest_DIR})

find_package(Threads REQUIRED)

# Compiler settings
set(CMAKE_C_COMPILER /usr/bin/clang)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)

# Enable testing and include GoogleTest module
include(CTest)
include(GoogleTest)
enable_testing()

# Find GTest package with forced paths
find_package(GTest REQUIRED PATHS ${GTEST_ROOT} ${GTest_DIR})
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
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    # If legacy BOOST_LIBRARY_THREAD is defined, use it; otherwise, use the imported target Boost::thread.
    $<$<BOOL:${BOOST_LIBRARY_THREAD}>:${BOOST_LIBRARY_THREAD}>
    $<$<NOT:$<BOOL:${BOOST_LIBRARY_THREAD}>>:Boost::thread>
)

# # Register test
# add_test(NAME thread_pool_tests COMMAND thread_pool_test)

# Enable test discovery
set_target_properties(thread_pool_test PROPERTIES
    GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST
)

gtest_discover_tests(thread_pool_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
