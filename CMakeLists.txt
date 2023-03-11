cmake_minimum_required(VERSION 3.14)

message(STATUS "Start of piconet CMakeLists.txt")

project(piconet
    VERSION 1.0.0
    DESCRIPTION "A primitive C++ neural network library."
    LANGUAGES CXX
#    HOMEPAGE "https://github.com/schuderer/piconet"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Provide a set of variables containing installation directories for various artifacts
include(GNUInstallDirs)


# When picograd is stable, replace this:
#add_library(picograd INTERFACE)
#target_include_directories(picograd INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/../picograd/include/")
# ...with this:
# https://medium.com/analytics-vidhya/c-dependency-management-with-cmakes-fetchcontent-4ceca4693a5d
include(FetchContent)
FetchContent_Declare(picograd
    GIT_REPOSITORY "https://github.com/schuderer/picograd"
    GIT_TAG "origin/lib_prep"  # or something like GIT_TAG "932a2ca50666138256dae56fbb16db3b1cae133a"
)
FetchContent_MakeAvailable(picograd)


add_library(${PROJECT_NAME} INTERFACE)  # because of header-only library, otherwise STATIC/SHARED (I think)

# Specify header locations
# From https://medium.com/swlh/c-project-structure-for-cmake-67d60135f6f5 :
target_include_directories(${PROJECT_NAME} INTERFACE  # would be PUBLIC for library with cpp files under src
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Add a subdirectory for the library source files
add_subdirectory(src)  # empty because of header-only library

# Ability to run experimental code in main.cpp
option(RUN_TRYOUT_MAIN "Build executable from main.cpp" ON)
if (RUN_TRYOUT_MAIN)
    set(TRYOUT_TARGET piconet_tryout)
    add_executable(${TRYOUT_TARGET} main.cpp)
    target_include_directories(${TRYOUT_TARGET} PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
    )
    # Compile options and linker flags
    # target_compile_options(${PROJECT_NAME} INTERFACE ...)
    target_link_libraries(${TRYOUT_TARGET} ${PROJECT_NAME} picograd)
    ## Installation destination
    install(TARGETS ${TRYOUT_TARGET}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

# Build unit tests if top-level project
option(VERBOSE "Turn on verbose logging for library builds" ON)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    if(VERBOSE)
        add_compile_definitions(VERBOSE_${PROJECT_NAME})
    endif()
    include(CTest)
    add_subdirectory(test)
endif()


