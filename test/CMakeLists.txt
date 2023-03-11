# https://stackoverflow.com/a/59060287
message(STATUS "Start of test/CMakeLists.txt")

find_package(GTest REQUIRED)
#include(GoogleTest)  # apparently not necessary

set(TEST_BINARY tests)

add_executable(${TEST_BINARY}
    "${CMAKE_CURRENT_SOURCE_DIR}/nn_test.cpp"
)
#target_include_directories(${TEST_BINARY} PRIVATE  # apparently not necessary
#    ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}>
#)

target_link_libraries(${TEST_BINARY} PRIVATE ${PROJECT_NAME} picograd GTest::GTest GTest::Main)
#gtest_discover_tests(tests)  # Not needed with CTest (and creates error due to additional tests_NOT_BUILD target)

add_test(
    NAME tests
    COMMAND ${TEST_BINARY}
)

message(STATUS "End of test/CMakeLists.txt")


